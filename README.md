Transfer learning architectures and their training strategies are evaluated on primary MIG Welding defect datasets.
A two step protocol was designed to first find out the best configurations (Phase-1) and then evaluate them (Phase-2) varying training strategies
Phase-1 used a combination of TensorFlow and PyTorch (For RegNet_y_1.6_GF and ConvNeXT_Small which was not available in TensorFlow). While Phase-2 used PyTorch exclusively.

Strategies involve: 1) Training Base Models 2) Fine-Tuning for 5 epochs and 3) SENet Integration

Package Version Used:
tensorflow 2.10.1
torch v2.4.0 (PyTorch)
torchvision v0.19.0
torchinfo v1.8.0
numpy v1.26.4
matplotlib v3.8.4

Data is available in: https://drive.google.com/drive/folders/1KXJ4bLZxGgEsjzJQTBiaq7Qe6K62Ae4w?usp=drive_link
