# Assignment 8: Continual Learning and Test in Production

In this assignment I simulate a model that must learn new classes after deployment. I compare three update strategies, then test the updated models the way companies do in production.

## Setup
- **Data:** CIFAR-10, reused from `../Assignment6/data`
  - **Subset A:** classes 0 to 4 (airplane, automobile, bird, cat, deer)
  - **Subset B:** classes 5 to 9 (dog, frog, horse, ship, truck)
  - Both subsets are balanced, about 4,500 training images per class.
- **Model:** a small VGG-style CNN (Convolutional Neural Network) with 10 outputs, trained on a GPU (Graphics Processing Unit).

## Results

| Method | Old classes (A) | New classes (B) | All 10 classes | Train time | Samples processed |
|---|---|---|---|---|---|
| Initial (A only) | 0.856 | 0.000 | 0.428 | – | – |
| Stateless retrain (A + B from scratch) | 0.771 | 0.855 | **0.813** | 28.9 s | 360,000 |
| Stateful naive (B only) | **0.000** | 0.910 | 0.455 | 7.7 s | 90,152 |
| Stateful + replay (B + 20% of A) | 0.510 | 0.890 | 0.700 | 9.0 s | 108,120 |

**Main findings:**
- Naive stateful training shows **catastrophic forgetting**: accuracy on the old classes fell to 0 after one epoch.
- Replay keeps part of the old knowledge at about one third of the cost of stateless retraining.
- **A/B test:** stateless 0.808 vs replay 0.708, z = -11.6, p ≈ 0, so stateless is significantly better.
- **ε-greedy bandit (ε = 0.1):** it sent 93% of traffic to the stateless model and reached 0.798 accuracy. Always using the best model would give 0.813, and random routing gives 0.656.

## Files
- `Assignment_8_Continual_Learning.ipynb`: code, outputs, interpretation of every result, and recommendations
- `initial_model.pth`: the initial model trained on Subset A
- `method_comparison.png`, `training_cost.png`, `bandit.png`: saved charts

## How to run
Open the notebook with the **Python 3.14 (Assignment6)** kernel (PyTorch with CUDA) and choose Run All. It takes about 5 to 10 minutes on a GPU. The CIFAR-10 data must already exist in `../Assignment6/data`.
