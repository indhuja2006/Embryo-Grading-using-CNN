import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ----- Page Configuration -----
st.set_page_config(page_title="Blastocyst Grading App", layout="wide")

# ----- Model Definition -----
class MultiOutputResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        base_model.fc = nn.Identity()
        self.base = base_model
        self.fc_exp = nn.Linear(512, 6)  # Expansion stages: 1 to 6
        self.fc_icm = nn.Linear(512, 4)  # Grades A to D
        self.fc_te = nn.Linear(512, 4)   # Grades A to D

    def forward(self, x):
        x = self.base(x)
        out_exp = self.fc_exp(x)
        out_icm = self.fc_icm(x)
        out_te = self.fc_te(x)
        return out_exp, out_icm, out_te

# ----- Load Model -----
@st.cache_resource
def load_model():
    model = MultiOutputResNet()
    model.load_state_dict(torch.load("multi_output_blastocyst_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# ----- Grad-CAM -----
def generate_gradcam(model, image_tensor):
    image_tensor.requires_grad = True
    feature_maps = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    hook = model.base.layer4.register_forward_hook(forward_hook)
    outputs = model(image_tensor)
    loss = sum([torch.max(out) for out in outputs])
    loss.backward()

    gradients = image_tensor.grad[0].detach().numpy()
    fmap = feature_maps[0].squeeze(0).detach().numpy()

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

# ----- Format Prediction -----
def format_prediction(exp_idx, icm_idx, te_idx):
    expansion = exp_idx + 1
    grade_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    return f"{expansion}{grade_map[icm_idx]}{grade_map[te_idx]}"

# ----- Biological Feedback -----
def get_biological_feedback(grade):
    expansion_feedback = {
        1: "Stage 1: Early blastocyst with minimal blastocoel cavity.",
        2: "Stage 2: Blastocoel cavity >50% of the embryo.",
        3: "Stage 3: Fully expanded blastocyst.",
        4: "Stage 4: Expanded with thinned zona pellucida.",
        5: "Stage 5: Hatching blastocyst.",
        6: "Stage 6: Fully hatched blastocyst."
    }
    icm_feedback = {
        "A": "Grade A ICM: Many tightly packed, well-defined cells (high viability).",
        "B": "Grade B ICM: Moderate, loosely grouped cells (good viability).",
        "C": "Grade C ICM: Few disorganized cells (lower viability).",
        "D": "Grade D ICM: Very few or degenerative cells (poor viability)."
    }
    te_feedback = {
        "A": "Grade A TE: Cohesive layer of many cells (ideal implantation).",
        "B": "Grade B TE: Fewer cells with minor gaps (adequate support).",
        "C": "Grade C TE: Sparse, disorganized cells (lower implantation).",
        "D": "Grade D TE: Fragmented or few cells (poor implantation)."
    }

    try:
        exp = int(grade[0])
        icm = grade[1]
        te = grade[2]
        return f"""
        ### Biological Feedback
        - *Expansion Stage:* {expansion_feedback.get(exp)}
        - *ICM Grade:* {icm_feedback.get(icm)}
        - *TE Grade:* {te_feedback.get(te)}
        """
    except:
        return "No detailed feedback available."

# ----- Generate PDF Report (No label or confidence) -----
def generate_pdf_report(expansion, icm, te, grade, feedback_text, original_img, gradcam_img):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 50, "Embryo Grading Report")

    # Section: Prediction Summary Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 90, "Prediction Summary")

    # Prediction content
    c.setFont("Helvetica", 12)
    y = height - 110
    c.drawString(70, y, f"Expansion Stage: {expansion}")
    c.drawString(70, y - 20, f"Inner Cell Mass Grade: {icm}")
    c.drawString(70, y - 40, f"Trophectoderm Grade: {te}")
    c.drawString(70, y - 60, f"Combined Grade: {grade}")

    # Section: Biological Feedback
    c.setFont("Helvetica-Bold", 14)
    y -= 90
    c.drawString(50, y, "Biological Feedback")

    # Feedback content
    c.setFont("Helvetica", 11)
    feedback_lines = feedback_text.strip().replace("### Biological Feedback", "").replace("*", "").replace("-", "•").splitlines()
    y -= 20
    for line in feedback_lines:
        if line.strip():
            c.drawString(70, y, line.strip())
            y -= 16

    # Section: Image Display Header
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Original Embryo Image & Grad-CAM Visualization")

    # Images
    img_y = y - 240
    orig = ImageReader(Image.fromarray(original_img))
    grad = ImageReader(Image.fromarray(gradcam_img))
    c.drawImage(orig, 60, img_y, width=220, height=220)
    c.drawImage(grad, 320, img_y, width=220, height=220)

    # Image captions
    c.setFont("Helvetica-BoldOblique", 11)
    c.drawCentredString(60 + 110, img_y - 15, "Original Embryo Image")
    c.drawCentredString(320 + 110, img_y - 15, "Grad-CAM Visualization")

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, 30, "Generated using Embryo AI Analyzer")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer



# ----- Image Transform -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----- Streamlit UI -----
st.title(" Embryo Grading App")
model = load_model()

uploaded_file = st.file_uploader("Upload blastocyst image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad = True

    # Prediction
    with torch.no_grad():
        exp_out, icm_out, te_out = model(input_tensor)

    pred_exp = torch.argmax(exp_out).item()
    pred_icm = torch.argmax(icm_out).item()
    pred_te = torch.argmax(te_out).item()

    grade_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    expansion = pred_exp + 1
    icm = grade_map[pred_icm]
    te = grade_map[pred_te]
    grade = f"{expansion}{icm}{te}"

    feedback = get_biological_feedback(grade)

    st.markdown(f"""
    ## 🧬 Prediction Summary
    - **Expansion Stage:** {expansion}  
    - **Inner Cell Mass Grade:** {icm}  
    - **Trophectoderm Grade:** {te}  
    - **Grade:** {grade}
    """)
    st.markdown(feedback)

    # Grad-CAM
    cam = generate_gradcam(model, input_tensor)
    cam_image = np.array(image.resize((224, 224)))
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cam_image, 0.6, heatmap, 0.4, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cam_image, caption="Original Embryo Image", use_column_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)

    # Generate and Download PDF Report
    pdf = generate_pdf_report(
        expansion, icm, te, grade, feedback,
        cam_image, overlay
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf,
        file_name=f"Embryo_Report_{grade}.pdf",
        mime="application/pdf"
    )
