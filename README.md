Project: GP-GNN Active Learning for Atomistic Modeling

Goal:
- Use SOAP-based Gaussian Process (GP) for uncertainty-aware exploration
- Train a Graph Neural Network (GNN) to replace GP for scalable MD
- Use DFT as ground-truth anchor via active learning

Key Ideas:
- GP provides mean + uncertainty (mu, sigma)
- GP uncertainty controls:
    - DFT acquisition
    - training weights
- GNN trained on:
    - DFT data (full weight)
    - GP data (uncertainty-weighted)

Workflow:
1. Run MD using GP (early stage)
2. Compute sigma for all frames
3. Select high-sigma points → DFT
4. Train GNN on DFT + weighted GP data
5. Transition to GNN-driven MD
6. Continue active learning with DFT updates
