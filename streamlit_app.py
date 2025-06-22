import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchvision import models
from gradcam import GradCAM
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import tempfile
import os

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

# Updated PDF generation
def generate_pdf(label, prob, grade, feedback, original_image, overlay_image):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.darkblue)
    c.drawCentredString(width / 2, height - 50, "Embryo Viability Assessment Report")

    c.setStrokeColor(colors.grey)
    c.setLineWidth(0.5)
    c.line(40, height - 60, width - 40, height - 60)

    # Prediction Info
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.black)
    c.drawString(50, height - 90, "Prediction Summary")

    c.setFont("Helvetica", 11)
    c.drawString(60, height - 110, f"Prediction: {label}")
    c.drawString(60, height - 130, f"Confidence Score: {prob:.4f}")
    c.drawString(60, height - 150, f"Confidence Grade: {grade}")

    # Feedback
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 180, "Biological Feedback")

    c.setFont("Helvetica", 11)
    lines = feedback.strip().split("\n")
    for i, line in enumerate(lines):
        c.drawString(60, height - 200 - i*15, line.strip())

    # Save images temporarily
    orig_path = os.path.join(tempfile.gettempdir(), "orig.jpg")
    gradcam_path = os.path.join(tempfile.gettempdir(), "gradcam.jpg")

    original_image.save(orig_path)
    Image.fromarray(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)).save(gradcam_path)

    # Draw Images
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 350, "Original Embryo Image & Grad-CAM Visualization")
    c.drawImage(orig_path, 60, height - 550, width=200, height=200)
    c.drawImage(gradcam_path, 300, height - 550, width=200, height=200)

    c.save()
    return temp_file.name

# Streamlit UI
st.set_page_config(page_title="Embryo Quality Classifier", layout="centered")
st.title("🧬 Embryo Quality Classifier with Grad-CAM 🔍")

uploaded_file = st.file_uploader("Upload an embryo image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Embryo Image", use_column_width=True)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        label = "GOOD" if prob > 0.5 else "NOT GOOD"

        grade = "Grade A" if prob >= 0.85 else "Grade B" if prob >= 0.7 else "Grade C" if prob >= 0.5 else "Uncertain"

    feedback = (
        "Blastomere structure appears well-organized with minimal fragmentation.\n"
        "Zona pellucida is intact and uniform, indicating optimal developmental potential.\n"
        "Cytoplasmic texture is smooth, consistent with high implantation success."
        if label == "GOOD"
        else
        "Blastomere fragmentation observed; may hinder development.\n"
        "Zona pellucida appears uneven or breached.\n"
        "Irregular cytoplasmic texture — lower implantation potential."
    )

    st.markdown(f"### 🔎 Prediction: **{label}** (Confidence: `{prob:.4f}`)")
    st.markdown(f"### 🏷️ AI-based Morphology Grade: **{grade}**")
    st.markdown("### 📋 Biological Feedback")
    st.markdown(feedback)

    # Grad-CAM
    st.markdown("### 🧠 Grad-CAM Visualization")
    heatmap = gradcam.generate(input_tensor)
    img = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
    st.image(overlay, caption="Highlighted Areas Influencing Decision", use_column_width=True)

    # PDF Report
    pdf_path = generate_pdf(label, prob, grade, feedback, image, overlay)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="embryo_report.pdf",
            mime="application/pdf"
        )
