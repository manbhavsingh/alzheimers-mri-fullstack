import base64
import io
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import CLASS_NAMES, GradCAM, build_model, inference_transform

WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "app/weights/best_resnet.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_grad_cam = None


def load_model():
    """Lazily load the model + Grad-CAM once per process."""
    global _model, _grad_cam

    if _model is not None:
        return _model, _grad_cam

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at '{WEIGHTS_PATH}'. Export "
            "best_resnet.pth from the training notebook and place it there "
            "(or set MODEL_WEIGHTS_PATH env var). See README for details."
        )

    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    grad_cam = GradCAM(model=model, target_layer=model.layer4)

    _model, _grad_cam = model, grad_cam
    return _model, _grad_cam


def _overlay_heatmap(pil_image: Image.Image, cam: np.ndarray) -> str:
    """Resize cam to original image size, colorize, and blend over the scan.

    Returns a base64-encoded PNG data URL of the overlay.
    """
    img_rgb = np.array(pil_image.convert("RGB").resize((224, 224)))

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.55 * img_rgb + 0.45 * heatmap)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _image_to_data_url(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").resize((224, 224)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def predict(image_bytes: bytes) -> dict:
    """Run classification + Grad-CAM on a single uploaded MRI scan."""
    model, grad_cam = load_model()

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("L")
    input_tensor = inference_transform(pil_image).unsqueeze(0).to(DEVICE)

    # Predicted class (separate no-grad pass; GradCAM.generate also runs
    # a forward pass with grad enabled for the CAM computation below)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))

    cam, _ = grad_cam.generate(input_tensor, class_idx=pred_idx)

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)
        },
        "original_image": _image_to_data_url(pil_image),
        "gradcam_overlay": _overlay_heatmap(pil_image, cam),
    }
