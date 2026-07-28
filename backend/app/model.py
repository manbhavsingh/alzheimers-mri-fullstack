"""
Model architecture and Grad-CAM implementation.

This MUST match the architecture used in the training notebook
(hack4health-alzheimer-mri-cnn.ipynb) exactly, or the saved weights
(best_resnet.pth) will fail to load / will load incorrectly.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented",
]

IMAGE_SIZE = 224

# Same normalization used in the training notebook (single-channel MRI)
inference_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
)


def build_model(num_classes: int = 4) -> nn.Module:
    """Recreate the exact architecture used during training."""
    model = models.resnet18(weights=None)

    # Grayscale MRI input (1 channel instead of 3)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Ported directly from the training notebook. Targets model.layer4,
    the final convolutional block, for class-discriminative heatmaps.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam, output
