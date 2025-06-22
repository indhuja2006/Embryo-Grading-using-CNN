# gradcam.py
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        pred = torch.sigmoid(output)
        pred.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max()

        return heatmap


def generate_gradcam(model, image_path, output_path, target_layer):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_cv = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(output_path, superimposed_img)
