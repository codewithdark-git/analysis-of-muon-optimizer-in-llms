#!/bin/bash

# Usage examples for the MoE training script with attention selection

echo "🚀 MoE Training with Attention Selection Examples"
echo "================================================="

# Example 1: Train with Multi-Head Self Attention (MHSA) - Default
echo "1️⃣ Training with MHSA (standard self-attention):"
python moe_training_script.py --attention mhsa --max_steps 1000 --batch_size 24

echo ""

# Example 2: Train with Multi-Head Latent Attention (MHLA)
echo "2️⃣ Training with MHLA (latent attention):"
python moe_training_script.py --attention mhla --num_latents 64 --max_steps 1000 --batch_size 24

echo ""

# Example 3: Train with MHLA using more latent tokens
echo "3️⃣ Training with MHLA (more latent tokens):"
python moe_training_script.py --attention mhla --num_latents 128 --max_steps 1000 --batch_size 16

echo ""

# Example 4: Compare both attention mechanisms
echo "4️⃣ Comparing both MHSA and MHLA:"
python moe_training_script.py --attention both --num_latents 64 --max_steps 500 --batch_size 16

echo ""

# Example 5: Longer training with more experts
echo "5️⃣ Longer training with more experts:"
python moe_training_script.py --attention mhla --num_latents 64 --num_experts 16 --max_steps 2000 --batch_size 16

echo ""

# Example 6: Quick test with fewer steps
echo "6️⃣ Quick test run:"
python moe_training_script.py --attention both --num_latents 32 --max_steps 200 --batch_size 8

echo ""
echo "📋 Available Arguments:"
echo "  --attention: 'mhsa', 'mhla', or 'both'"
echo "  --num_latents: Number of latent tokens for MHLA (default: 64)"
echo "  --num_experts: Number of experts in MoE (default: 8)"
echo "  --max_steps: Maximum training steps (default: 1000)"
echo "  --batch_size: Batch size (default: 24)"
echo ""
echo "💡 Tips:"
echo "  - Use 'both' to compare MHSA vs MHLA performance"
echo "  - Reduce batch_size if you get out of memory errors"
echo "  - MHLA is more memory efficient for long sequences"
echo "  - More latent tokens = higher quality but more computation"
