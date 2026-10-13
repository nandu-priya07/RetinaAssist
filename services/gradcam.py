"""Explainable Retina Lens: Grad-CAM for the saved nested EfficientNetB0 model.

Approach (mirrors the tested Colab method):
  1. Build a sub-model on the nested `efficientnetb0` backbone that returns both
     the `top_conv` feature maps and the backbone output in ONE call. Keeping both
     outputs inside the backbone graph avoids Keras 3 graph-connectivity errors.
  2. Re-apply the saved classification head (pooling -> dropout -> dense) manually
     inside a GradientTape.
  3. Weight the `top_conv` maps by their pooled gradients.

Grad-CAM shows regions that influenced the model's output. It is NOT proof that
a lesion or disease is present.
"""
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from services.predictor import get_model, preprocess

BACKBONE_NAME = "efficientnetb0"
CONV_LAYER_NAME = "top_conv"

_grad_parts = None


def _build_parts():
    global _grad_parts
    if _grad_parts is None:
        import keras

        model = get_model()
        backbone = model.get_layer(BACKBONE_NAME)
        conv_layer = backbone.get_layer(CONV_LAYER_NAME)
        backbone_grad = keras.Model(backbone.inputs, [conv_layer.output, backbone.output])

        # Head layers that come after the backbone in the saved model.
        idx = model.layers.index(backbone)
        head = [layer for layer in model.layers[idx + 1:]]
        _grad_parts = (backbone_grad, head)
    return _grad_parts


def compute_heatmap(img, class_index):
    """Return a 2-D Grad-CAM map (values 0-1) at the conv-layer resolution."""
    backbone_grad, head = _build_parts()
    x = tf.convert_to_tensor(preprocess(img))

    with tf.GradientTape() as tape:
        conv_out, features = backbone_grad(x, training=False)
        tape.watch(conv_out)
        y = features
        for layer in head:
            y = layer(y, training=False)
        class_score = y[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    if grads is None:
        raise RuntimeError("Gradient computation failed for Grad-CAM.")

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.nn.relu(cam).numpy()
    max_val = cam.max()
    if max_val <= 0 or not np.isfinite(max_val):
        # No positive evidence for the class: return an all-zero map rather than noise.
        return np.zeros_like(cam, dtype=np.float32)
    return (cam / max_val).astype(np.float32)


def render(img, cam, max_width=800, alpha=0.4):
    """Return (original, heatmap, overlay) as PIL RGB images at the same display size."""
    original = img.convert("RGB")
    if original.width > max_width:
        h = round(original.height * max_width / original.width)
        original = original.resize((max_width, h), Image.LANCZOS)
    w, h = original.size

    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    orig_arr = np.asarray(original, dtype=np.float32)
    overlay = np.clip(orig_arr * (1 - alpha) + heat_rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return original, Image.fromarray(heat_rgb), Image.fromarray(overlay)


def generate(img, class_index):
    """Compute Grad-CAM for `class_index` and return (original, heatmap, overlay) PIL images."""
    cam = compute_heatmap(img, class_index)
    return render(img, cam)
