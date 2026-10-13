"""RetinaGrade AI: image validation, preprocessing and inference with the saved model.

The selected model (fine-tuned EfficientNetB0) contains its own Rescaling and
Normalization layers inside the nested `efficientnetb0` backbone, so it expects
224x224 RGB float pixels in the 0-255 range. Do NOT divide by 255 here.
"""
import io
import os
import threading

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, UnidentifiedImageError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "retinaassist_efficientnet_finetuned.keras")

IMG_SIZE = 224
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION = 128
MAX_DIMENSION = 8000

# Guard against decompression-bomb images.
Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION

_model = None
_model_lock = threading.Lock()


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def allowed_extension(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image(file_bytes, filename=None):
    """Validate raw upload bytes and return an RGB PIL image."""
    if filename is not None and not allowed_extension(filename):
        raise ImageValidationError("Unsupported file type. Please upload a PNG or JPEG image.")
    if not file_bytes:
        raise ImageValidationError("The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise ImageValidationError("The file is larger than 10 MB.")

    try:
        with Image.open(io.BytesIO(file_bytes)) as probe:
            fmt = probe.format
            probe.verify()  # detects truncated / corrupted files
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The file could not be decoded as an image. It may be corrupted.") from exc

    if fmt not in ("PNG", "JPEG"):
        raise ImageValidationError(f"Unsupported image format ({fmt}). Please upload a PNG or JPEG image.")

    width, height = img.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image is too small ({width}x{height}). Minimum size is {MIN_DIMENSION}x{MIN_DIMENSION} pixels."
        )
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageValidationError(f"Image is too large ({width}x{height}). Maximum side is {MAX_DIMENSION} pixels.")

    return img.convert("RGB")


def preprocess(img):
    """Resize to 224x224 and return a float32 batch of shape (1, 224, 224, 3), values 0-255."""
    resized = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def get_model():
    """Load the saved Keras model once and reuse it."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
                import keras

                _model = keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def predict(img):
    """Run the real model on a PIL image and return grade, label and all five class scores."""
    batch = preprocess(img)
    scores = get_model().predict(batch, verbose=0)[0].astype(float)
    if scores.shape != (len(CLASS_NAMES),):
        raise RuntimeError(f"Unexpected model output shape: {scores.shape}")

    order = np.argsort(scores)[::-1]
    top, second = int(order[0]), int(order[1])
    return {
        "grade": top,
        "label": CLASS_NAMES[top],
        "scores": [float(s) for s in scores],
        "top_score": float(scores[top]),
        "second_grade": second,
        "second_score": float(scores[second]),
        "margin": float(scores[top] - scores[second]),
    }
