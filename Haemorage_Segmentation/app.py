# import gradio as gr
# import numpy as np
# import cv2
# import torch
# from model import build_unet

# # Constants - must match training dimensions
# H = 340  # Height (rows)
# W = 512  # Width (columns)
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load model
# model = build_unet()
# model.load_state_dict(torch.load("files/checkpoint.pth", map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# def predict(image):
#     """Process input image and generate segmentation mask"""
#     # Convert RGB to BGR (OpenCV format)
#     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     # Resize to model's expected input size
#     image_resized = cv2.resize(image_bgr, (W, H))
    
#     # Normalize and prepare tensor
#     x = image_resized.astype(np.float32) / 255.0
#     x = np.transpose(x, (2, 0, 1))  # Change to CHW
#     x = np.expand_dims(x, axis=0)    # Add batch dimension
#     x = torch.from_numpy(x).to(DEVICE)

#     # Prediction
#     with torch.no_grad():
#         pred = model(x)
#         pred = torch.sigmoid(pred)
#         pred = pred[0].cpu().numpy()  # Remove batch dim

#     # Process mask
#     mask = (pred > 0.5).astype(np.uint8)[0] * 255  # Remove channel dim & threshold
#     mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
#     # Resize outputs to original dimensions for better visualization
#     original_display = cv2.resize(image_bgr, (W, H))
#     mask_display = cv2.resize(mask_bgr, (W, H))
    
#     # Create composite image (original | mask)
#     separator = np.ones((H, 10, 3), dtype=np.uint8) * 128
#     composite = np.hstack([original_display, separator, mask_display])
    
#     # Convert back to RGB for Gradio
#     return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

# # Gradio interface
# interface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="numpy", label="Retinal Image"),
#     outputs=gr.Image(type="numpy", label="Hemorrhage Segmentation"),
#     title="Retinal Hemorrhage Segmentation",
#     description="Upload a retinal fundus image to detect hemorrhagic regions",
#     allow_flagging="never"
# )

# if __name__ == "__main__":
#     interface.launch(server_port=7860, share=False)



import gradio as gr
import numpy as np
import cv2
import torch
from model import build_unet

# Constants - must match training
H, W = 340, 512  # Height, Width
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = build_unet()
model.load_state_dict(torch.load("files/checkpoint.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(image):
    """Process image and create terminal-style output"""
    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize to model input dimensions
    image_resized = cv2.resize(image_bgr, (W, H))
    
    # Prepare tensor
    x = image_resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW format
    x = np.expand_dims(x, axis=0)    # Add batch dimension
    x = torch.from_numpy(x).to(DEVICE)

    # Prediction
    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        pred = pred[0][0].cpu().numpy()  # Get first channel as numpy array

    # Create hemorrhage mask (white on black)
    mask = (pred > 0.5).astype(np.uint8) * 255
    
    # Convert to 3-channel for display
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Create components with exact terminal style
    original_display = image_resized.copy()
    separator = np.ones((H, 10, 3), dtype=np.uint8) * 128  # Gray line
    
    # Create composite: original | separator | prediction
    composite = np.hstack([original_display, separator, mask_display])
    
    # Convert back to RGB for Gradio
    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

# Gradio interface with exact terminal output style
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Retinal Image"),
    outputs=gr.Image(type="numpy", label="Hemorrhage Segmentation"),
    title="Retinal Hemorrhage Segmentation",
    description=(
        "Upload a retinal fundus image. "
        "Output shows: Original Image | Predicted Hemorrhage Mask"
    ),
    examples=[
        ["path/to/IDRiD_55.png"]  # Add example images
    ]
)

if __name__ == "__main__":
    interface.launch(server_port=7860, share=True)