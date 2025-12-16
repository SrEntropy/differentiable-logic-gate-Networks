# NPC25-difflogic-4-control
NPC25 Exploring Differential Logic Networks for Control
workshop:  [Telluride Neuromophs](https://sites.google.com/view/telluride-2025/home)

Papers to read:
1) [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
# Differentiable Logic Gate Networks

## Overview
This project builds on the ideas from [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277), exploring how Boolean logic operations can be represented in continuous, differentiable space.  
By making logic gates trainable with gradient descent, we aim to bridge symbolic reasoning and deep learning.

---

## Motivation
Traditional neural networks excel at pattern recognition but struggle with explicit logical reasoning.  
Differentiable logic gates provide:
- A way to embed symbolic rules into neural architectures
- Residual learning capacity for hybrid models
- Potential applications in neuromorphic AI and event‑driven vision

---

## Dataset & Testing
- **MNIST (event‑driven images)** was used as the initial benchmark to validate the architecture.  
- The network achieved strong classification accuracy while demonstrating the feasibility of differentiable logic gate learning.  
- Next step: integration with a **real event‑driven camera**, extending the approach to neuromorphic hardware.

---

## Features
- Differentiable AND, OR, XOR, and other Boolean gates  
- Gradient‑based training for logical functions  
- Conversion utilities for exporting trained networks to Verilog  
- Event‑driven dataset compatibility  

---

## Purpose
This repo serves as a **research playground** for experimenting with differentiable logic gates.  
It is designed for students, researchers, and practitioners interested in:
- Neuromorphic computing  
- Symbolic‑connectionist hybrid models  
- Event‑driven vision and real‑time AI systems  

---

## Prerquisites

This repo uses the uv package manager, see the [uv repo](https://github.com/astral-sh/uv/) for installation instructions. The most likely you will have just to do:
```bash
pip install uv
```

## How to run?
The script will run as is with default settings, no config needed.
```bash
uv run mnist.py
```

### Expected result after running for 5 minutes
The script trains MNIST classifier to ~97.75% **binarized** test accuracy roughly in 5 minutes on MacBookPro with Apple silicon.
```
MNIST_515382 EPOCH=25/30     TST            acc=97.65%, test_acc_diff= 2.30%, loss=0.074
Iteration       5460 - Loss  0.010 - Pass (%) 12 35 | Conn (%) 72 100 | e-∇x10 -75 -70 -68 | Logits 219..611 μ382±59.4/11.5 = ±5.1 ‖1692‖
Iteration       5670 - Loss  0.015 - Pass (%) 12 35 | Conn (%) 72 100 | e-∇x10 -74 -70 -68 | Logits 244..595 μ383±59.4/11.5 = ±5.1 ‖1698‖
Iteration       5880 - Loss  0.011 - Pass (%) 12 35 | Conn (%) 72 100 | e-∇x10 -73 -70 -67 | Logits 238..611 μ382±60.5/11.5 = ±5.2 ‖1697‖
Iteration       6090 - Loss  0.014 - Pass (%) 12 35 | Conn (%) 72 100 | e-∇x10 -73 -68 -66 | Logits 216..592 μ382±59.3/11.5 = ±5.1 ‖1693‖
Iteration       6300 - Loss  0.009 - Pass (%) 12 35 | Conn (%) 72 100 | e-∇x10 -75 -70 -69 | Logits 243..596 μ382±59.3/11.5 = ±5.1 ‖1695‖
MNIST_515382 EPOCH=30/30     TRN loss=0.011 acc=99.95%
MNIST_515382 EPOCH=30/30 BIN TRN            acc=99.95%, train_acc_diff=0.00%
MNIST_515382 EPOCH=30/30     TST            acc=97.74%, test_acc_diff= 2.21%, loss=0.074
Training took 359.38 seconds, per iteration: 57.04 milliseconds
    TEST loss=0.074 acc=97.74%
BIN TEST loss=0.075 acc=97.74%
```

## How to convert trained network to Verilog?

```bash
uv run pth_to_npz.py INSERT_PTH_FILENAME_HERE
```
---
## 📜 License
MIT License. Free to use, modify, and share.

