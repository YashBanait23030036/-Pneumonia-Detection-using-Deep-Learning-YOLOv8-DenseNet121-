import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import hashlib
from ultralytics import YOLO
from pipeline_code.model import get_densenet121

# Cache predictions using image hash
prediction_cache = {}

def get_image_hash(img: Image.Image) -> str:
    img_bytes = img.tobytes()
    return hashlib.md5(img_bytes).hexdigest()

st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩺 Pneumonia Detection")
st.markdown("Upload a chest X-ray to detect **pneumonia** using **YOLOv8** and **DenseNet121**.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_hash = get_image_hash(image)

    if img_hash in prediction_cache:
        st.info("✅ This image was already analyzed. Loaded cached result.")
        final_result, yolo_detected, densenet_pred = prediction_cache[img_hash]
    else:
        with st.spinner("🔍 Running YOLOv8 detection..."):
            yolo_model = YOLO("D:/rsna_dataset/runs/detect/train/weights/best.pt")
            results = yolo_model.predict([np.array(image)], save=False, conf=0.11)
            yolo_result = results[0]
            boxes = yolo_result.boxes.xyxy.cpu().numpy() if yolo_result.boxes else []
            yolo_detected = len(boxes) > 0

            if yolo_detected:
                yolo_img = yolo_result.plot()
                st.image(yolo_img, caption="YOLOv8 Detection", use_container_width=True)
            else:
                st.warning("⚠️ No pneumonia region detected by YOLOv8.")

        with st.spinner("🧠 Running DenseNet121 classification..."):
            densenet_model = get_densenet121(num_classes=2)
            checkpoint = torch.load("D:/rsna_dataset/checkpoints/best_model.pth", map_location="cpu")
            densenet_model.load_state_dict(checkpoint, strict=False)
            densenet_model.eval()

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            img_tensor = preprocess(image).unsqueeze(0)
            output = densenet_model(img_tensor)
            _, pred = torch.max(output, 1)
            classes = ['Normal', 'Pneumonia']
            densenet_pred = classes[pred.item()]

        # Final decision: Pneumonia if any model detects it
        if yolo_detected or densenet_pred == "Pneumonia":
            final_result = "Pneumonia"
        else:
            final_result = "Normal"

        prediction_cache[img_hash] = (final_result, yolo_detected, densenet_pred)

    st.markdown("---")
    st.subheader("📋 Final Result:")
    st.success(f"**{final_result}**")

    with st.expander("🔎 Model Insights"):
        st.write(f"YOLOv8 Detection: {'Yes' if yolo_detected else 'No'}")
        st.write(f"DenseNet121 Prediction: **{densenet_pred}**")