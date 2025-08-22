# import gradio as gr
# import numpy as np
# import cv2
# import torch
# import os
# from model import build_unet

# # Constants
# H, W = 512, 512  # Image dimensions
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load model
# model = build_unet()
# model.load_state_dict(torch.load("files/checkpoint.pth", map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# # Path to your test dataset for ground truth masks
# TEST_MASK_DIR = "new_data/test/mask/"

# def predict(image, filename):
#     """Process image and create terminal-style output with three sections"""
#     # Convert RGB to BGR (OpenCV format)
#     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     # Resize to model input dimensions
#     image_resized = cv2.resize(image_bgr, (W, H))
    
#     # Prepare tensor
#     x = image_resized.astype(np.float32) / 255.0
#     x = np.transpose(x, (2, 0, 1))  # CHW format
#     x = np.expand_dims(x, axis=0)    # Add batch dimension
#     x = torch.from_numpy(x).to(DEVICE)

#     # Prediction
#     with torch.no_grad():
#         pred = model(x)
#         pred = torch.sigmoid(pred)
#         pred = pred[0][0].cpu().numpy()  # Get first channel as numpy array

#     # Create blood vessel mask (white vessels on black background)
#     pred_mask = (pred > 0.5).astype(np.uint8) * 255
    
#     # Try to find ground truth mask if available
#     gt_mask = None
#     try:
#         mask_path = os.path.join(TEST_MASK_DIR, filename)
#         if os.path.exists(mask_path):
#             gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             gt_mask = cv2.resize(gt_mask, (W, H))
#     except:
#         pass
    
#     # Create components
#     original_display = image_resized.copy()
#     pred_mask_display = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    
#     # Create separator lines
#     separator = np.ones((H, 10, 3), dtype=np.uint8) * 128
    
#     # Create composite based on whether ground truth is available
#     if gt_mask is not None:
#         # Process ground truth mask
#         gt_mask_display = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
        
#         composite = np.hstack([
#             original_display, 
#             separator, 
#             gt_mask_display, 
#             separator, 
#             pred_mask_display
#         ])
#     else:
#         composite = np.hstack([
#             original_display, 
#             separator, 
#             pred_mask_display
#         ])
    
#     # Convert back to RGB for Gradio
#     return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

# # Gradio interface with filename input
# with gr.Blocks(title="Retinal Blood Vessel Segmentation") as demo:
#     gr.Markdown("# Retinal Blood Vessel Segmentation")
#     gr.Markdown("Upload a retinal fundus image or select a test image")
    
#     with gr.Row():
#         with gr.Column():
#             image_input = gr.Image(type="numpy", label="Retinal Image")
#             filename_input = gr.Textbox(
#                 label="Mask Filename (e.g., 21_test_0.png)", 
#                 value="21_test_0.png"
#             )
#             run_button = gr.Button("Run Segmentation")
        
#         with gr.Column():
#             output_image = gr.Image(type="numpy", label="Segmentation Result")
    
#     gr.Examples(
#         examples=[
#             ["../new_data/test/image/21_test_0.png", "21_test_0.png"],
#             ["../new_data/test/image/22_test_0.png", "22_test_0.png"]
#         ],
#         inputs=[image_input, filename_input],
#         outputs=output_image,
#         fn=predict,
#         cache_examples=True
#     )
    
#     run_button.click(
#         fn=predict,
#         inputs=[image_input, filename_input],
#         outputs=output_image
#     )

# if __name__ == "__main__":
#     demo.launch(server_port=7860, share=True)

# import gradio as gr
# import numpy as np
# import cv2
# import torch
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, accuracy_score
# from model import build_unet  # your model building function

# # Constants
# H, W = 512, 512  # Image dimensions
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load model
# model = build_unet()
# model.load_state_dict(torch.load("files/checkpoint.pth", map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# # Path to your test dataset for ground truth masks
# TEST_MASK_DIR = "new_data/test/mask/"

# def calculate_metrics(pred_mask, gt_mask):
#     # Flatten masks to 1D arrays
#     pred_flat = (pred_mask > 127).astype(int).flatten()
#     gt_flat = (gt_mask > 127).astype(int).flatten()

#     # Compute metrics
#     metrics = {
#         "Jaccard": jaccard_score(gt_flat, pred_flat),
#         "F1": f1_score(gt_flat, pred_flat),
#         "Recall": recall_score(gt_flat, pred_flat),
#         "Precision": precision_score(gt_flat, pred_flat),
#         "Accuracy": accuracy_score(gt_flat, pred_flat),
#     }
#     return metrics

# def plot_metrics(metrics):
#     fig, ax = plt.subplots(figsize=(5, 3))
#     names = list(metrics.keys())
#     values = list(metrics.values())
    
#     ax.bar(names, values, color='skyblue')
#     ax.set_ylim(0, 1)
#     ax.set_title("Segmentation Metrics")
    
#     for i, v in enumerate(values):
#         ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
#     plt.tight_layout()
#     return fig

# def predict(image, filename):
#     if image is None or filename.strip() == "":
#         return None, None
    
#     # If image is a filepath string, load it
#     if isinstance(image, str):
#         image = cv2.imread(image)
#         if image is None:
#             return None, None
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Convert RGB to BGR for OpenCV processing
#     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     # Resize to model input dimensions
#     image_resized = cv2.resize(image_bgr, (W, H))
    
#     # Prepare tensor
#     x = image_resized.astype(np.float32) / 255.0
#     x = np.transpose(x, (2, 0, 1))  # CHW format
#     x = np.expand_dims(x, axis=0)    # Add batch dimension
#     x = torch.from_numpy(x).to(DEVICE)

#     # Prediction
#     with torch.no_grad():
#         pred = model(x)
#         pred = torch.sigmoid(pred)
#         pred = pred[0][0].cpu().numpy()  # Get first channel as numpy array

#     # Create blood vessel mask (white vessels on black background)
#     pred_mask = (pred > 0.5).astype(np.uint8) * 255
    
#     # Try to find ground truth mask if available
#     gt_mask = None
#     try:
#         mask_path = os.path.join(TEST_MASK_DIR, filename)
#         if os.path.exists(mask_path):
#             gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             gt_mask = cv2.resize(gt_mask, (W, H))
#     except Exception as e:
#         print("Error loading ground truth mask:", e)
    
#     # Prepare images for display
#     original_display = image_resized.copy()
#     pred_mask_display = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
#     separator = np.ones((H, 10, 3), dtype=np.uint8) * 128
    
#     if gt_mask is not None:
#         gt_mask_display = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
#         composite = np.hstack([
#             original_display,
#             separator,
#             gt_mask_display,
#             separator,
#             pred_mask_display
#         ])
#         # Calculate and plot metrics
#         metrics = calculate_metrics(pred_mask, gt_mask)
#         metric_fig = plot_metrics(metrics)
#     else:
#         composite = np.hstack([
#             original_display,
#             separator,
#             pred_mask_display
#         ])
#         metric_fig = None

#     composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
#     return composite_rgb, metric_fig

# # Gradio interface
# with gr.Blocks(title="Retinal Blood Vessel Segmentation with Metrics") as demo:
#     gr.Markdown("# Retinal Blood Vessel Segmentation with Metrics")
#     gr.Markdown("Upload a retinal fundus image or select a test image")
    
#     with gr.Row():
#         with gr.Column():
#             image_input = gr.Image(type="filepath", label="Retinal Image")
#             filename_input = gr.Textbox(
#                 label="Mask Filename (e.g., 21_test_0.png)", 
#                 value="21_test_0.png"
#             )
#             run_button = gr.Button("Run Segmentation")
        
#         with gr.Column():
#             output_image = gr.Image(type="numpy", label="Segmentation Result")
#             metrics_plot = gr.Plot(label="Segmentation Metrics")
    
#     gr.Examples(
#         examples=[
#             ["../new_data/test/image/21_test_0.png", "21_test_0.png"],
#             ["../new_data/test/image/22_test_0.png", "22_test_0.png"]
#         ],
#         inputs=[image_input, filename_input],
#         outputs=[output_image, metrics_plot],
#         fn=predict,
#         cache_examples=False
#     )
    
#     run_button.click(
#         fn=predict,
#         inputs=[image_input, filename_input],
#         outputs=[output_image, metrics_plot]
#     )

# if __name__ == "__main__":
#     demo.launch(server_port=7860, share=True)

import gradio as gr
import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, accuracy_score
from model import build_unet  # your model building function

# Constants
H, W = 512, 512  # Image dimensions
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = build_unet()
model.load_state_dict(torch.load("files/checkpoint.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict_mask(image):
    """Predict segmentation mask given an input image"""
    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image_bgr, (W, H))
    x = image_resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x).to(DEVICE)
    
    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)[0,0].cpu().numpy()
    
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
    return pred_mask_rgb

def calculate_metrics(pred_mask, gt_mask):
    pred_bin = (pred_mask > 127).flatten()
    gt_bin = (gt_mask > 127).flatten()
    metrics = {
        "Jaccard": jaccard_score(gt_bin, pred_bin),
        "F1 Score": f1_score(gt_bin, pred_bin),
        "Recall": recall_score(gt_bin, pred_bin),
        "Precision": precision_score(gt_bin, pred_bin),
        "Accuracy": accuracy_score(gt_bin, pred_bin),
    }
    return metrics

def predict_with_metrics(image, gt_mask):
    """Predict mask and calculate metrics given input image and ground truth mask"""
    # Resize ground truth mask
    gt_mask_resized = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    
    pred_mask_rgb = predict_mask(image)
    pred_mask_gray = cv2.cvtColor(pred_mask_rgb, cv2.COLOR_RGB2GRAY)
    
    metrics = calculate_metrics(pred_mask_gray, gt_mask_resized)
    
    # Stack original image, GT mask, and predicted mask side by side for display
    image_resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (W, H))
    gt_mask_rgb = cv2.cvtColor(gt_mask_resized, cv2.COLOR_GRAY2BGR)
    
    separator = np.ones((H, 10, 3), dtype=np.uint8) * 128
    composite = np.hstack([image_resized, separator, gt_mask_rgb, separator, cv2.cvtColor(pred_mask_rgb, cv2.COLOR_RGB2BGR)])
    composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    
    return composite_rgb, metrics

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Retinal Blood Vessel Segmentation Demo")
    
    with gr.Tab("Segment Only"):
        gr.Markdown("Upload an image to see the predicted segmentation mask.")
        img_input = gr.Image(type="numpy", label="Retinal Image")
        pred_output = gr.Image(label="Predicted Segmentation Mask")
        btn1 = gr.Button("Run Segmentation")
        btn1.click(fn=predict_mask, inputs=img_input, outputs=pred_output)
    
    with gr.Tab("Segment + Metrics"):
        gr.Markdown("Upload an image and the corresponding ground truth mask to see segmentation and metrics.")
        img_input2 = gr.Image(type="numpy", label="Retinal Image")
        gt_input = gr.Image(type="numpy", label="Ground Truth Mask", image_mode='L')  # grayscale mask
        pred_output2 = gr.Image(label="Composite Output: Original | GT Mask | Prediction")
        metrics_output = gr.Label(label="Segmentation Metrics")
        btn2 = gr.Button("Run Segmentation & Metrics")
        btn2.click(fn=predict_with_metrics, inputs=[img_input2, gt_input], outputs=[pred_output2, metrics_output])
        
if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
