import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io
import pandas as pd
import base64
import os

st.set_page_config(page_title="FedIP_MobileNetV2 — DR Demo", layout="wide")

# Pillow LANCZOS fallback
try:
    resample = Image.Resampling.LANCZOS
except:
    resample = Image.LANCZOS

# -------------------------
# Helper functions
# -------------------------
def preprocess_pil_image(pil_img, target_size=(224, 224)):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = ImageOps.fit(pil_img, target_size, resample)
    arr = img_to_array(pil_img) / 255.0
    return arr

def batch_predict(model, images_arr):
    preds = model.predict(images_arr, batch_size=16, verbose=0)
    return preds

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        shape = getattr(layer.output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, class_index):
    img_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)
    last_conv = get_last_conv_layer(model)
    conv_layer = model.get_layer(last_conv)

    heatmap_model = tf.keras.models.Model([model.inputs],
                                          [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = heatmap_model(img_tensor)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (224, 224))
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).convert("L")
    return heatmap_img

def overlay_heatmap_on_image(img_pil, heatmap_pil):
    heatmap_color = heatmap_pil.convert("RGB").resize(img_pil.size)
    overlay = Image.blend(img_pil, heatmap_color, alpha=0.4)
    return overlay

def to_download_link(df, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download predictions CSV</a>'
    return href

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")

default_model = "models/FedIP_MobileNetV2.h5"  # ✅ Relative path for Streamlit Cloud
model_path = st.sidebar.text_input("Model Path (.h5)", value=default_model)

class_list = ["No_DR","Mild","Moderate","Severe","Proliferative_DR"]
class_names = st.sidebar.text_area("Class labels (comma-sep)", value=",".join(class_list)).split(",")

gradcam = st.sidebar.checkbox("Enable Grad-CAM", False)
max_imgs = st.sidebar.slider("Max Images to Display", 1, 60, 30)

@st.cache_resource
def load_model_cached(path):
    return load_model(path)

try:
    model = load_model_cached(model_path)
    st.sidebar.success("✅ Model Loaded")
except Exception as e:
    model = None
    st.sidebar.error(f"❌ Model Load Error: {e}")

# -------------------------
# Main UI
# -------------------------
st.title("🩺 DR Classification — FedIP MobileNetV2")
st.write("Upload fundus images to classify diabetic retinopathy stages.")

files = st.file_uploader("Upload fundus images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if files and model:
    files = files[:max_imgs]

    st.info(f"Processing {len(files)} image(s)…")

    imgs, pila, names = [], [], []

    for f in files:
        image = Image.open(io.BytesIO(f.read()))
        pila.append(image.copy())
        imgs.append(preprocess_pil_image(image))
        names.append(f.name)

    imgs = np.stack(imgs, axis=0)

    st.write("Running inference...")
    preds = batch_predict(model, imgs)

    idxs = np.argmax(preds, axis=1)
    confs = np.max(preds, axis=1)
    labels = [class_names[i] for i in idxs]

    df = pd.DataFrame({
        "filename": names,
        "prediction": labels,
        "confidence": confs
    })

    st.markdown(to_download_link(df), unsafe_allow_html=True)
    st.write("---")

    cols = st.columns(3)

    for i,(pil_img, label, conf, idx) in enumerate(zip(pila, labels, confs, idxs)):
        col = cols[i % 3]
        with col:
            st.markdown(f"**{names[i]}**")
            st.write(f"Prediction: **{label}**")
            st.write(f"Confidence: **{conf:.2f}**")

            disp = ImageOps.fit(pil_img.convert("RGB"), (300,300), resample)
            st.image(disp)

            if gradcam:
                heatmap = make_gradcam_heatmap(preprocess_pil_image(pil_img), model, idx)
                overlay = overlay_heatmap_on_image(disp, heatmap)
                st.write("Grad-CAM:")
                st.image(overlay)

            st.write("---")

    st.success("✅ Completed")

else:
    st.info("Upload images to start.")
