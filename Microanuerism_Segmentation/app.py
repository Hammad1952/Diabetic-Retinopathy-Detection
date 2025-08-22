import gradio as gr
import torch
import numpy as np
import cv2
from model import build_unet
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, accuracy_score

# Constants
H, W = 512, 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Microaneurysm model
model = build_unet()
model.load_state_dict(torch.load("files/checkpoint.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict_mask(image):
    """Predict only mask from image"""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image_bgr, (W, H))
    x = image_resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    pred_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
    return pred_rgb

def evaluate_metrics(pred, gt):
    pred_bin = (pred > 127).flatten()
    gt_bin = (gt > 127).flatten()
    return {
        "Jaccard": round(jaccard_score(gt_bin, pred_bin), 4),
        "F1 Score": round(f1_score(gt_bin, pred_bin), 4),
        "Recall": round(recall_score(gt_bin, pred_bin), 4),
        "Precision": round(precision_score(gt_bin, pred_bin), 4),
        "Accuracy": round(accuracy_score(gt_bin, pred_bin), 4),
    }

def predict_and_evaluate(image, gt_mask):
    """Predict mask and show side-by-side with GT + metrics"""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image_bgr, (W, H))
    gt_resized = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    x = image_resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    pred_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
    gt_rgb = cv2.cvtColor(gt_resized, cv2.COLOR_GRAY2RGB)

    sep = np.ones((H, 10, 3), dtype=np.uint8) * 128
    composite = np.hstack([image_resized, sep, gt_rgb, sep, pred_rgb])
    composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

    metrics = evaluate_metrics(pred_mask, gt_resized)
    return composite_rgb, metrics

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Microaneurysm Segmentation")

    with gr.Tab("🖼️ Predict Mask Only"):
        with gr.Row():
            img_input = gr.Image(type="numpy", label="Upload Fundus Image")
        with gr.Row():
            mask_output = gr.Image(label="Predicted Microaneurysm Mask")
        with gr.Row():
            btn1 = gr.Button("Predict Mask")
            btn1.click(fn=predict_mask, inputs=img_input, outputs=mask_output)

    with gr.Tab("📊 Predict + Evaluate with Ground Truth"):
        with gr.Row():
            img_input2 = gr.Image(type="numpy", label="Upload Fundus Image")
            gt_input2 = gr.Image(type="numpy", label="Upload Ground Truth Mask", image_mode='L')
        with gr.Row():
            composite_output = gr.Image(label="Input | Ground Truth | Prediction")
        with gr.Row():
            metric_output = gr.Label(label="Evaluation Metrics")
        with gr.Row():
            btn2 = gr.Button("Predict & Evaluate")
            btn2.click(fn=predict_and_evaluate, inputs=[img_input2, gt_input2], outputs=[composite_output, metric_output])

# Run app
if __name__ == "__main__":
    demo.launch(server_port=7862, share=True)
