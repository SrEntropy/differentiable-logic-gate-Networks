# npc25-difflogic-4-control
NPC25 Exploring Differential Logic Networks for Control

Papers to read:
1) [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)

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
