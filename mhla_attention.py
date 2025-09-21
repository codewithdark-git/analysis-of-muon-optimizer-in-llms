import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torchtune.modules import RotaryPositionalEmbeddings


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=10000)

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MHLA) - Uses learned latent queries that attend to input tokens
    and then cross-attend back to produce outputs. This reduces computational complexity
    from O(n²) to O(n×k) where k is the number of latent tokens.
    
    Based on concepts from Perceiver and similar latent attention architectures.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        max_seq_len: int,
        num_latents: int = 64,  # Number of latent tokens
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.num_latents = num_latents
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Learned latent tokens
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        
        # First attention: latents attend to input (cross-attention)
        self.q_latent = nn.Linear(d_model, d_model, bias=False)  # For latent queries
        self.kv_input = nn.Linear(d_model, d_model * 2, bias=False)  # For input keys and values
        
        # Second attention: input attends to processed latents (cross-attention)
        self.q_input = nn.Linear(d_model, d_model, bias=False)  # For input queries
        self.kv_latent = nn.Linear(d_model, d_model * 2, bias=False)  # For latent keys and values
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary positional embeddings
        self.rotary = Rotary(self.d_k, max_seq_len)
        
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Expand latent tokens for batch
        latents = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_latents, d_model]
        
        # === First Attention: Latents attend to Input ===
        # Latent queries
        Q_latent = self.q_latent(latents).reshape(
            batch_size, self.num_latents, self.n_heads, self.d_k
        ).transpose(1, 2)  # [B, H, num_latents, D]
        
        # Input keys and values
        kv_input = self.kv_input(x).reshape(
            batch_size, seq_len, 2, self.n_heads, self.d_k
        ).permute(2, 0, 3, 1, 4)
        K_input, V_input = kv_input[0], kv_input[1]  # [B, H, T, D]
        
        # Apply RoPE to keys (latent queries don't need positional encoding)
        K_input = self.rotary(K_input.transpose(1, 2)).transpose(1, 2)
        
        # Cross-attention: latents attend to input
        latent_attn_output = F.scaled_dot_product_attention(
            Q_latent, K_input, V_input,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0
        )  # [B, H, num_latents, D]
        
        # Reshape latent attention output
        latent_attended = latent_attn_output.transpose(1, 2).reshape(
            batch_size, self.num_latents, self.d_model
        )  # [B, num_latents, d_model]
        
        # === Second Attention: Input attends to processed Latents ===
        # Input queries
        Q_input = self.q_input(x).reshape(
            batch_size, seq_len, self.n_heads, self.d_k
        ).transpose(1, 2)  # [B, H, T, D]
        
        # Apply RoPE to input queries
        Q_input = self.rotary(Q_input.transpose(1, 2)).transpose(1, 2)
        
        # Latent keys and values (from attended latents)
        kv_latent = self.kv_latent(latent_attended).reshape(
            batch_size, self.num_latents, 2, self.n_heads, self.d_k
        ).permute(2, 0, 3, 1, 4)
        K_latent, V_latent = kv_latent[0], kv_latent[1]  # [B, H, num_latents, D]
        
        # Cross-attention: input attends to latents
        output_attn = F.scaled_dot_product_attention(
            Q_input, K_latent, V_latent,
            dropout_p=self.dropout if self.training else 0.0
        )  # [B, H, T, D]
        
        # Reshape and project output
        output = output_attn.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self Attention for comparison"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2] # [B, H, T, D]

        # Apply RoPE on [B, T, H, D]
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


def create_attention_layer(
    attention_type: str,
    d_model: int, 
    n_heads: int, 
    max_seq_len: int,
    dropout: float = 0.1,
    num_latents: int = 64
) -> nn.Module:
    """
    Factory function to create attention layers
    
    Args:
        attention_type: "mhsa" for Multi-Head Self Attention or "mhla" for Multi-Head Latent Attention
        d_model: Model dimension
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        num_latents: Number of latent tokens (only used for MHLA)
    
    Returns:
        Attention layer module
    """
    if attention_type.lower() == "mhsa":
        return MultiHeadSelfAttention(d_model, n_heads, max_seq_len, dropout)
    elif attention_type.lower() == "mhla":
        return MultiHeadLatentAttention(d_model, n_heads, max_seq_len, num_latents, dropout)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}. Use 'mhsa' or 'mhla'")


if __name__ == "__main__":
    # Test both attention mechanisms
    batch_size, seq_len, d_model = 2, 128, 384
    n_heads = 8
    max_seq_len = 512
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("Testing Multi-Head Self Attention (MHSA):")
    mhsa = create_attention_layer("mhsa", d_model, n_heads, max_seq_len)
    mhsa_params = sum(p.numel() for p in mhsa.parameters())
    
    with torch.no_grad():
        mhsa_out = mhsa(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {mhsa_out.shape}")
        print(f"  Parameters: {mhsa_params:,}")
        print(f"  Computational complexity: O(n²) where n={seq_len}")
    
    print("\nTesting Multi-Head Latent Attention (MHLA):")
    num_latents = 64
    mhla = create_attention_layer("mhla", d_model, n_heads, max_seq_len, num_latents=num_latents)
    mhla_params = sum(p.numel() for p in mhla.parameters())
    
    with torch.no_grad():
        mhla_out = mhla(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {mhla_out.shape}")
        print(f"  Parameters: {mhla_params:,}")
        print(f"  Latent tokens: {num_latents}")
        print(f"  Computational complexity: O(n×k) where n={seq_len}, k={num_latents}")
    
    print(f"\nParameter comparison:")
    print(f"  MHSA parameters: {mhsa_params:,}")
    print(f"  MHLA parameters: {mhla_params:,}")
    print(f"  Difference: {mhla_params - mhsa_params:+,} ({((mhla_params/mhsa_params - 1)*100):+.1f}%)")
    
    print(f"\nComputational complexity comparison (for seq_len={seq_len}):")
    mhsa_ops = seq_len * seq_len
    mhla_ops = seq_len * num_latents * 2  # Two cross-attention operations
    print(f"  MHSA operations: ~{mhsa_ops:,} (n²)")
    print(f"  MHLA operations: ~{mhla_ops:,} (2×n×k)")
    print(f"  Efficiency gain: {mhsa_ops/mhla_ops:.1f}x fewer operations")
