The Muon Optimizer: A New Frontier in Training Large Language Models

A groundbreaking optimization technique, Muon, is challenging the long-standing dominance of AdamW in training large language models (LLMs). Spearheaded by researchers including Jeremy Bernstein and Keller Jordan, and prominently featured in the work of Moonshot AI's Kimi team, Muon promises significant improvements in computational efficiency and training stability, paving the way for more powerful and accessible LLMs.

The Muon optimizer, at its core, is a matrix-aware method that leverages a technique called Newton-Schulz orthogonalization.[1][2] Unlike traditional optimizers that adjust individual weights, Muon operates on entire matrices of parameters, which form the fundamental building blocks of modern neural networks.[1] This approach helps to ensure that all parts of the network learn more equally, preventing a few dominant components from monopolizing the learning process.[1] The result is a more efficient and stable training process, particularly for the massive models that are becoming the industry standard.[3][4]

The Minds Behind Muon: Credits and Contributions

The development of the Muon optimizer was a collaborative effort. Key contributors to its conception and refinement include:

Jeremy Bernstein: A pivotal figure in deriving the theoretical principles behind Muon.[1] His work on what he terms "metricizing neural networks" — assigning distance measures to the internal spaces of a network — laid the groundwork for a more principled approach to optimization.[5]

Keller Jordan: Instrumental in the practical implementation and empirical validation of Muon.[2][6] His work has demonstrated Muon's effectiveness in setting new speed records in training benchmarks.[6][7]

The Kimi Team at Moonshot AI: This team, including researchers like Jingyuan Liu and Jianlin Su, authored the influential paper "Muon is Scalable for LLM Training."[8] Their work was crucial in demonstrating that Muon could be effectively scaled to train large-scale language models, a previously unproven capability.[8] They also introduced "Moonlight," a 16-billion parameter Mixture-of-Experts (MoE) model trained using Muon.[8]

Other Contributors: Researchers such as Laker Newhouse, Vlado Boza, Yuchen Jin, Jiacheng You, and Franz Cesista have also been recognized for their significant contributions to the development and understanding of Muon.[1][9]

Deconstructing Muon: The Mathematical Blocks

The innovation of Muon lies in its approach to updating the weight matrices of a neural network. It can be understood through the following key steps:

Metrizing the Linear Layer: The process begins by defining measures of size for the inputs, weights, and outputs of a linear layer.[1]

Perturbing the Layer: The optimizer then considers how changes to the weight matrices will affect the layer's output.[1]

Dualizing the Gradient: A crucial step is to select a weight update that maximizes the improvement in the loss function while keeping the change in the output's behavior within a defined bound.[1]

Fast Dualization with Newton-Schulz: The final and most computationally innovative step is the use of the Newton-Schulz iteration to efficiently perform the gradient orthogonalization.[1][2] This method effectively replaces the standard update with the nearest semi-orthogonal matrix, ensuring a more stable and efficient learning process.[6]

In practice, Muon is often used as a specialized tool for the hidden layers of a neural network, while a more traditional optimizer like AdamW is employed for other parameters such as embeddings and biases.[4]

The Moonshot AI "Kimi" Paper: Scaling Muon for LLMs

The paper "Muon is Scalable for LLM Training" from Moonshot AI's Kimi team marked a significant milestone in the adoption of the Muon optimizer.[8] Prior to this work, Muon's effectiveness had been demonstrated on smaller-scale models, but its scalability to the massive datasets and parameter counts of modern LLMs was an open question.[8]

The Kimi team identified and addressed two critical challenges to scaling Muon:

The Addition of Weight Decay: They found that incorporating a standard AdamW-style weight decay mechanism was essential to prevent the weights and layer outputs from growing excessively and harming the model's performance.

Per-Parameter Update Scale Adjustment: Careful adjustment of the update scale for each parameter allowed Muon to work effectively in large-scale training without the need for extensive hyperparameter tuning.

Their experiments demonstrated that Muon could achieve approximately twice the computational efficiency of the widely used AdamW optimizer.[8] This translates to significant savings in the time and resources required to train state-of-the-art models.[3] The successful training of their "Moonlight" model, a powerful 16B-parameter MoE model, served as concrete proof of Muon's capabilities at scale.[8]

More recent advancements from Moonshot AI in their Kimi K2 model have introduced MuonClip, a hybrid optimizer that combines Muon with a technique called QK-Clip. This innovation addresses the issue of "exploding attention scores" during training, a major cause of instability in large transformers. MuonClip has enabled the stable training of models with up to a trillion parameters.

Sources
help
jeremybernste.in
huggingface.co
arxiv.org
medium.com
youtube.com
github.io
google.com
arxiv.org
lakernewhouse.com
Google Search Suggestions
Display of Search Suggestions is required when using Grounding with Google Search. Learn more
Muon optimizer "Jeremy"
Muon optimizer development team
Keller Jordan Muon optimizer collaborators