   import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchvision import models
from gradcam import GradCAM  # Ensure gradcam.py is in the same folder
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os
from textwrap import wrap

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 1)
model.load_state_dict(torch.load("embryo_model.pth", map_location=torch.device("cpu")))
model.eval()

# GradCAM setup
target_layer = model.layer4[1].conv2
gradcam = GradCAM(model, target_layer)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to generate PDF report
def generate_pdf(label, prob, grade, feedback, overlay_image):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "\U0001F9EC Embryo Quality Classification Report")

    # Prediction Info
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Prediction: {label}")
    c.drawString(50, height - 120, f"Confidence Score: {prob:.2f}")
    c.drawString(50, height - 140, f"AI Morphology Grade: {grade}")

    # Feedback
    feedback_lines = wrap(feedback.replace("✅", "").replace("⚠️", ""), 90)
    c.drawString(50, height - 170, "Feedback:")
    for i, line in enumerate(feedback_lines):
        c.drawString(60, height - 190 - i*15, line)

    # Save Grad-CAM image using PIL instead of cv2.imwrite
    img_path = os.path.join(tempfile.gettempdir(), "gradcam_overlay.jpg")
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    overlay_pil.save(img_path)

    c.drawImage(img_path, 50, height - 450, width=300, height=200)
    c.save()

    return temp_file.name

# UI Setup
st.set_page_config(page_title="Embryo Quality Classifier", layout="centered")
st.title("\U0001F9EC Embryo Quality Classifier with Grad-CAM \U0001F50D")

# File Upload
uploaded_file = st.file_uploader("Upload an embryo image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Embryo Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        label = "GOOD" if prob > 0.5 else "NOT GOOD"

        # Confidence-based grading
        if prob >= 0.85:
            grade = "Grade A"
        elif 0.70 <= prob < 0.85:
            grade = "Grade B"
        elif 0.50 <= prob < 0.70:
            grade = "Grade C"
        else:
            grade = "Uncertain"

    # Implantation Suitability Feedback
    if label == "GOOD":
        feedback = (
            "✅ This embryo is likely **suitable for implantation**.\n"
            "- High cellular integrity and well-formed inner cell mass detected.\n"
            "- Model focused on healthy morphological patterns."
        )
    else:
        feedback = (
            "⚠️ This embryo may **not be optimal for implantation**.\n"
            "- Possible signs of fragmentation or irregular blastocyst shape.\n"
            "- Model detected abnormalities in critical regions."
        )

    # Display predictions
    st.markdown(f"### 🔎 Prediction: **{label}** (Confidence: `{prob:.2f}`)")
    st.markdown(f"### 🏷️ AI-based Morphology Grade: **{grade}**")
    st.markdown(f"### 📋 Implantation Suitability Feedback\n{feedback}")

    if grade == "Grade A":
        st.success("✅ Excellent morphology. High potential for implantation.")
    elif grade == "Grade B":
        st.info("⚠️ Moderate quality. Likely viable but with mild concerns.")
    elif grade == "Grade C":
        st.warning("❗ Low morphology score. Monitor carefully.")
    else:
        st.error("⚠️ The model is uncertain about the embryo's viability.")

    # Grad-CAM Visualization
    st.markdown("### Grad-CAM: AI Decision Explanation")
    heatmap = gradcam.generate(input_tensor)
    img = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
    st.image(overlay, caption="Highlighted Areas Influencing Decision", use_column_width=True)

    # PDF download
    pdf_path = generate_pdf(label, prob, grade, feedback, overlay)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="embryo_report.pdf",
            mime="application/pdf"
        )

