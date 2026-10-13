"""Stage 1 check: load the saved model and print its structure and shapes."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "retinaassist_efficientnet_finetuned.keras")

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded:", MODEL_PATH)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

print("\nTop-level layers:")
for layer in model.layers:
    print(f"  {layer.name:30s} {layer.__class__.__name__}")

# Check whether the nested EfficientNet backbone has its own Rescaling/Normalization
# layers. If it does, the model expects raw 0-255 pixels (no /255 needed).
backbone = next((l for l in model.layers if isinstance(l, keras.Model)), None)
if backbone is not None:
    print(f"\nNested backbone: {backbone.name}")
    prep = [l.name for l in backbone.layers[:6]]
    print("  First backbone layers:", prep)
    print("  Has 'top_conv' layer:", any(l.name == "top_conv" for l in backbone.layers))

dummy = np.random.uniform(0, 255, size=(1, 224, 224, 3)).astype("float32")
scores = model.predict(dummy, verbose=0)[0]
print("\nDummy prediction (random noise input, meaningless values):")
print("  Scores:", np.round(scores, 4))
print("  Sum of scores:", round(float(scores.sum()), 4), "(about 1.0 means softmax output)")

assert model.input_shape == (None, 224, 224, 3), "Unexpected input shape"
assert model.output_shape == (None, 5), "Unexpected output shape"
print("\nSTAGE 1 CHECK PASSED")
