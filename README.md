# Concrete Crack Detection with ResNet18 (PyTorch)

Binary image classification (**crack / no crack** in concrete surfaces) using **transfer learning** with a pre-trained ResNet18 in PyTorch.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-torchvision-red) ![Task](https://img.shields.io/badge/Task-Image%20Classification-green)

## Results

| Metric | Score |
|---|---|
| Validation accuracy | **99.43%** |
| Training | 1 epoch, only the final layer trained |

The notebook also plots the training loss and shows the first misclassified validation images.

## Dataset

- **40,000 concrete surface images**: 20,000 with cracks (positive) and 20,000 without cracks (negative)
- Provided as pre-processed PyTorch tensors (IBM Skills Network, *AI Capstone Project with Deep Learning*)

## Method

1. **Data loading**: custom PyTorch `Dataset` class reading image tensors, split into training and validation sets
2. **Transfer learning**:
   - Load `resnet18` pre-trained on ImageNet
   - **Freeze** all convolutional layers (`requires_grad = False`)
   - Replace the final layer with `nn.Linear(512, 2)` for the two classes
3. **Training**: cross-entropy loss, Adam optimizer (learning rate 0.001), batch size 100
4. **Evaluation**: accuracy on the validation set, loss curve, analysis of misclassified samples

## How to run

```bash
git clone https://github.com/mohamedhadouu/resnet18-crack-detection-pytorch.git
cd resnet18-crack-detection-pytorch
pip install torch torchvision numpy pandas matplotlib pillow h5py tqdm
jupyter notebook 4_1_resnet18_PyTorch.ipynb
```

The first cells download the dataset (about 4.4 GB). A GPU is recommended.

## Tech stack

Python · PyTorch · torchvision · NumPy · Matplotlib · Pillow

## Next steps

- Fine-tune the last ResNet blocks instead of only the classifier
- Compare with other backbones (ResNet50, EfficientNet, MobileNet)
- Add Grad-CAM visualizations to show where the model sees the crack

## Author

**Mohamed Hadou**, AI / ML Engineer · [LinkedIn](https://www.linkedin.com/in/mohamed-hadou/) · [GitHub](https://github.com/mohamedhadouu)
