from reportlab.pdfgen import canvas
import io
import os
import base64
from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from gradcam import GradCAM
from torchvision import models
import torch.nn as nn


# Create Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming binary classification (Good / Not Good)
model.load_state_dict(torch.load('embryo_model.pth', map_location=torch.device('cpu')))
model.eval()

labels = ['NOT GOOD', 'GOOD']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return render_template('index.html', error="No file uploaded.")

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    label = labels[pred_class]

    # Grad-CAM
    target_layer = model.layer4[1].conv2
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    original_np = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_img, 0.4, 0)

    _, buffer = cv2.imencode('.png', overlay)
    gradcam_img = io.BytesIO(buffer.tobytes())
    gradcam_base64 = base64.b64encode(gradcam_img.getvalue()).decode()

    feedback = "Focus is on inner cell mass, which appears healthy." if label == 'GOOD' else "Trophectoderm appears less optimal for implantation."

    global latest_prediction
    latest_prediction = {
        'label': label,
        'confidence': f"{confidence*100:.2f}%",
        'feedback': feedback
    }

    return render_template('index.html',
                           prediction=True,
                           label=label,
                           prob=f"{confidence*100:.2f}%",
                           feedback=feedback,
                           gradcam_img=gradcam_base64)

@app.route('/download')
def download_report():
    if not latest_prediction:
        return "No prediction available to generate report.", 400

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.setFont("Helvetica", 14)
    c.drawString(100, 800, "Embryo Quality Prediction Report")
    c.drawString(100, 770, f"Prediction: {latest_prediction['label']}")
    c.drawString(100, 750, f"Confidence: {latest_prediction['confidence']}")
    c.drawString(100, 730, f"Feedback: {latest_prediction['feedback']}")
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="report.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    latest_prediction = {}
    app.run(debug=True)



