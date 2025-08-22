from datetime import datetime
import os
import torch
import gradio as gr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms as T
# Fallback for sklearn import issue
try:
    from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, accuracy_score
except ImportError:
    # Define fallback implementations
    def jaccard_score(y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / union if union != 0 else 0.0
        
    def f1_score(y_true, y_pred):
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
        fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        return 2 * (precision * recall) / (precision + recall + 1e-7)
        
    def recall_score(y_true, y_pred):
        tp = np.logical_and(y_true, y_pred).sum()
        fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()
        return tp / (tp + fn + 1e-7)
        
    def precision_score(y_true, y_pred):
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
        return tp / (tp + fp + 1e-7)
        
    def accuracy_score(y_true, y_pred):
        correct = np.equal(y_true, y_pred).sum()
        return correct / y_true.size

from src.model import DRModel
import importlib.util
import sys
import tempfile
import shutil
import pandas as pd
from PIL import Image
import json
from fpdf import FPDF
import google.generativeai as genai
import traceback
from fpdf import FPDF, FPDF_VERSION
import zipfile
import shutil
import tempfile
import base64

# Set environment variables to reduce memory usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Gemini initialization with detailed error handling
try:
    # Verify API key format
    API_KEY = ""
    if not API_KEY.startswith("AIza") or len(API_KEY) < 30:
        raise ValueError("Invalid API key format")
    
    # Configure Gemini
    genai.configure(api_key=API_KEY)
    print("✅ Gemini configured successfully")
    
    # Create the model instance
    gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")
    print("✅ Gemini model created successfully")
    
except Exception as e:
    print(f"❌ Gemini initialization failed: {str(e)}")
    print(traceback.format_exc())
    gemini_model = None

# Fallback recommendations
FALLBACK_RECOMMENDATIONS = {
    0: "No diabetic retinopathy detected. Continue annual eye exams and maintain good blood sugar control.",
    1: "Mild non-proliferative diabetic retinopathy detected. Schedule follow-up in 6-12 months. Improve blood sugar control.",
    2: "Moderate non-proliferative diabetic retinopathy detected. Schedule follow-up in 3-6 months. Tighten blood sugar and blood pressure control.",
    3: "Severe non-proliferative diabetic retinopathy detected. Refer to ophthalmologist immediately. May require laser treatment.",
    4: "Proliferative diabetic retinopathy detected. Urgent referral to ophthalmologist required. Likely needs laser treatment or surgery."
}

# === Load DR Classification Model ===
try:
    from src.model import DRModel
    DR_CHECKPOINT_PATH = "artifacts/dr-model.ckpt"
    model = DRModel.load_from_checkpoint(DR_CHECKPOINT_PATH, map_location="cpu")
    model.eval()
    print("✅ DR classification model loaded")
except Exception as e:
    print(f"❌ Failed to load DR model: {str(e)}")
    print(traceback.format_exc())
    model = None


# === Simplified Segmentation Model Loading ===
def load_segmentation_model(model_path, checkpoint_path):
    """Load segmentation model from file path"""
    # Load the model module
    spec = importlib.util.spec_from_file_location("seg_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the build_unet function
    if hasattr(module, 'build_unet'):
        build_unet = module.build_unet
    else:
        raise AttributeError(f"No build_unet function found in {model_path}")
    
    # Create and load the model
    model = build_unet()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# === Path Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Print directory structure for debugging
print("\n=== Current Directory Structure ===")
print(f"BASE_DIR: {BASE_DIR}")
print("Contents of BASE_DIR:")
for item in os.listdir(BASE_DIR):
    print(f"  - {item}")

# Corrected Segmentation Configuration
SEGMENTATION_CONFIG = {
    "vessels": {
        "model_path": os.path.join(BASE_DIR, "Blood_Vessel_Segmentation", "UNET", "model.py"),
        "checkpoint": os.path.join(BASE_DIR, "Blood_Vessel_Segmentation", "UNET", "files", "checkpoint.pth")
    },
    "hemorrhages": {
        "model_path": os.path.join(BASE_DIR, "Haemorage_Segmentation", "model.py"),
        "checkpoint": os.path.join(BASE_DIR, "Haemorage_Segmentation", "files", "checkpoint.pth")
    },
    "microaneurysms": {
        "model_path": os.path.join(BASE_DIR, "Microanuerism_Segmentation", "model.py"),
        "checkpoint": os.path.join(BASE_DIR, "Microanuerism_Segmentation", "files", "checkpoint.pth")
    },
    "exudates": {
        "model_path": os.path.join(BASE_DIR, "Hard_Exudate_Segmentation", "model.py"),
        "checkpoint": os.path.join(BASE_DIR, "Hard_Exudate_Segmentation", "files", "checkpoint.pth")
    },
    "optic_disc": {
        "model_path": os.path.join(BASE_DIR, "Optical_Disc_Segmentation", "model.py"),
        "checkpoint": os.path.join(BASE_DIR, "Optical_Disc_Segmentation", "files", "checkpoint.pth")
    },
    "macula": {
        "model_path": os.path.join(BASE_DIR, "Soft_Exudate_Segmentation", "model.py"),
        "checkpoint": os.path.join(BASE_DIR, "Soft_Exudate_Segmentation", "files", "checkpoint.pth")
    }
}

# Print paths for debugging
print("\n=== Path Verification ===")
for name, paths in SEGMENTATION_CONFIG.items():
    print(f"\n{name.upper()} model:")
    print(f"  Model path: {paths['model_path']}")
    print(f"  Exists: {os.path.exists(paths['model_path'])}")
    print(f"  Checkpoint path: {paths['checkpoint']}")
    print(f"  Exists: {os.path.exists(paths['checkpoint'])}")
print("=======================\n")

# === Load All Segmentation Models ===
seg_models = {}
for name, paths in SEGMENTATION_CONFIG.items():
    try:
        if os.path.exists(paths["model_path"]) and os.path.exists(paths["checkpoint"]):
            seg_models[name] = load_segmentation_model(paths["model_path"], paths["checkpoint"])
            print(f"✅ Loaded segmentation model for: {name}")
        else:
            print(f"⚠️ Missing files for {name}: {paths['model_path']}, {paths['checkpoint']}")
            seg_models[name] = None
    except Exception as e:
        print(f"❌ Failed to load {name}: {str(e)}")
        seg_models[name] = None

# === Define model_sizes, transform, labels, colors ===
model_sizes = {
    'vessels': (512, 512), 
    'hemorrhages': (340, 512), 
    'microaneurysms': (512, 512),
    'exudates': (512, 512), 
    'optic_disc': (512, 512), 
    'macula': (512, 512)
}
labels = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
label_colors = {
    "No DR": "#4CAF50", "Mild": "#8BC34A", "Moderate": "#FFC107",
    "Severe": "#FF9800", "Proliferative DR": "#F44336"
}
transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Update Prediction and Utility Functions ===
def predict_segmentation(image, model, size):
    """Predict segmentation mask for a single model with proper preprocessing"""
    # Convert PIL image to numpy array (RGB)
    image_np = np.array(image)
    
    # Convert to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Resize to model size
    image_resized = cv2.resize(image_bgr, size)
    
    # Normalize and convert to tensor
    x = image_resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # to CHW
    x = np.expand_dims(x, axis=0)   # add batch dimension
    x = torch.from_numpy(x)
    
    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()
    
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    return cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

def run_all_segmentations(image):
    """Run all available segmentation models"""
    results = {}
    for name, model in seg_models.items():
        if model is None:
            # Create placeholder image
            size = model_sizes.get(name, (512, 512))
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            cv2.putText(img, "Model Not Loaded", (50, 256), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            results[name] = img
            continue
            
        try:
            size = model_sizes.get(name, (512, 512))
            results[name] = predict_segmentation(image, model, size)
        except Exception as e:
            print(f"❌ Error in {name} segmentation: {e}")
            # Create placeholder image
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            cv2.putText(img, f"Error: {e}", (50, 256), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            results[name] = img
    return results

def create_bar_chart(confidences):
    plt.figure(figsize=(8, 4))
    names = list(confidences.keys())
    values = [confidences[n] for n in names]
    colors = [label_colors[n] for n in names]
    bars = plt.barh(names, values, color=colors)
    plt.xlim(0, 1)
    plt.xlabel('Confidence')
    plt.title('DR Prediction Probabilities')
    plt.gca().invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center')
    return plt.gcf()



    # Calculate metrics with error handling
    try:
        return {
            "Jaccard": jaccard_score(gt_bin, pred_bin),
            "F1 Score": f1_score(gt_bin, pred_bin),
            "Recall": recall_score(gt_bin, pred_bin),
            "Precision": precision_score(gt_bin, pred_bin),
            "Accuracy": accuracy_score(gt_bin, pred_bin)
        }
    except Exception as e:
        print(f"Metric calculation error: {str(e)}")
        return {k: 0.0 for k in ["Jaccard", "F1 Score", "Recall", "Precision", "Accuracy"]}
def create_composite(original, gt, pred, size):
    original_resized = cv2.resize(original, size)
    gt_resized = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
    
    # Convert to RGB if needed
    if len(gt_resized.shape) == 2:
        gt_rgb = cv2.cvtColor(gt_resized, cv2.COLOR_GRAY2BGR)
    else:
        gt_rgb = gt_resized
    
    separator = np.ones((size[1], 10, 3), dtype=np.uint8) * 128
    composite = np.hstack([
        cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR),
        separator, 
        gt_rgb, 
        separator, 
        pred
    ])
    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

# Add this function for Gemini recommendations
def generate_dr_recommendation(dr_stage, lesion_flags):
    """Generate clinical recommendations using Gemini Pro"""
    if gemini_model is None:
        return FALLBACK_RECOMMENDATIONS.get(dr_stage, "Recommendation service unavailable")
    
    stage_descriptions = {
        0: "No Diabetic Retinopathy",
        1: "Mild Non-proliferative Diabetic Retinopathy",
        2: "Moderate Non-proliferative Diabetic Retinopathy",
        3: "Severe Non-proliferative Diabetic Retinopathy",
        4: "Proliferative Diabetic Retinopathy"
    }
    
    prompt = f"""
    **Role**: You are an ophthalmology specialist AI assistant providing evidence-based recommendations for diabetic retinopathy.
    
    **Patient Diagnosis**:
    - DR Stage: {stage_descriptions[dr_stage]} (Class {dr_stage})
    - Lesion Presence:
        • Blood Vessels: {'Present' if lesion_flags['vessels'] else 'Absent'}
        • Hemorrhages: {'Present' if lesion_flags['hemorrhages'] else 'Absent'}
        • Microaneurysms: {'Present' if lesion_flags['microaneurysms'] else 'Absent'}
        • Hard Exudates: {'Present' if lesion_flags['exudates'] else 'Absent'}
        • Optic Disc: {'Present' if lesion_flags['optic_disc'] else 'Absent'}
        • Soft Exudates: {'Present' if lesion_flags['macula'] else 'Absent'}
    
    **Output Format**:
    1. 🧪 **Recommended Clinical Tests**
    2. 💊 **Treatment Options** 
    3. 📅 **Follow-up Schedule**
    4. 🥗 **Lifestyle Recommendations**
    5. ⚠️ **Risk Assessment**
    """
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=800,
                top_p=0.8
            )
        )
        return response.text
    except Exception as e:
        print(f"⚠️ Recommendation generation failed: {str(e)}")
        return FALLBACK_RECOMMENDATIONS.get(dr_stage, "Recommendation generation failed")


def predict_dr_and_segmentations(image):
    # DR Classification
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(input_tensor)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in labels}
    
    fig = create_bar_chart(confidences)
    
    # Run segmentations
    seg_results = run_all_segmentations(image)
    
    # Determine lesion presence
    lesion_flags = {}
    for name in model_sizes.keys():
        seg_mask = seg_results.get(name)
        if seg_mask is not None:
            if len(seg_mask.shape) == 3:
                seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_RGB2GRAY)
            non_zero = np.count_nonzero(seg_mask)
            lesion_flags[name] = (non_zero / seg_mask.size) > 0.005
    
    # Get DR stage
    max_class = max(confidences, key=confidences.get)
    dr_stage = [k for k, v in labels.items() if v == max_class][0]
    
    # Generate recommendation
    recommendation = generate_dr_recommendation(dr_stage, lesion_flags)
    
    # Return outputs in fixed order
    return (
        confidences,  # For output_label
        fig,          # For plot_output
        seg_results['vessels'],
        seg_results['hemorrhages'],
        seg_results['microaneurysms'],
        seg_results['exudates'],
        seg_results['optic_disc'],
        seg_results['macula'],
        recommendation  # For recommendation_output
    )

def predict_with_metrics(image, 
                         gt_vessels, gt_hemorrhages, gt_microaneurysms,
                         gt_exudates, gt_optic_disc, gt_macula):
    """
    Predict DR and run all segmentation models.
    Then compare predicted masks with provided ground truth masks and compute metrics.
    """
    # Step 1: DR Classification
    confidences, fig = predict_dr_and_segmentations(image)[:2]
    
    # Step 2: Run All Segmentations
    seg_results = run_all_segmentations(image)
    
    # Step 3: Ground truth masks
    gt_masks = [
        gt_vessels, gt_hemorrhages, gt_microaneurysms,
        gt_exudates, gt_optic_disc, gt_macula
    ]
    lesion_names = list(model_sizes.keys())

    composites = []
    metrics_list = []

    for i, name in enumerate(lesion_names):
        pred_mask = seg_results.get(name, np.zeros((model_sizes[name][1], model_sizes[name][0], 3), dtype=np.uint8))
        size = model_sizes.get(name, (512, 512))
        gt_mask = gt_masks[i] if i < len(gt_masks) else None

        # Convert to grayscale for metrics calculation
        if len(pred_mask.shape) == 3 and pred_mask.shape[2] == 3:
            pred_gray = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2GRAY)
        else:
            pred_gray = pred_mask

        # Handle ground truth mask
        if gt_mask is not None:
            gt_mask = np.array(gt_mask)
        else:
            gt_mask = np.zeros((size[1], size[0]), dtype=np.uint8)

        # Calculate metrics
        metrics = calculate_metrics(pred_gray, gt_mask)
        metrics_list.append(metrics)

        # Create composite overlay
        composite = create_composite(
            np.array(image), 
            gt_mask, 
            pred_mask,
            size
        )
        composites.append(composite)

    return [confidences, fig] + composites + metrics_list

def run_analysis(image):
    """Wrapper function to run analysis and store results"""
    results = predict_dr_and_segmentations(image)
    global latest_results
    
    # Store all results for export
    latest_results = {
        "image": image,
        "confidences": results[0],
        "plot_fig": results[1],
        "segmentations": [
            results[2],  # vessels
            results[3],  # hemorrhages
            results[4],  # microaneurysms
            results[5],  # exudates
            results[6],  # optic_disc
            results[7]   # macula
        ],
        "recommendation": results[8]  # recommendation
    }
    
    # Return confidence values for each stage
    confidences = results[0]
    return (
        results[0],  # confidences for output_label
        results[1],  # plot_fig
        *results[2:8],  # segmentation images
        results[8],  # recommendation
        *[confidences.get(stage, 0.0) for stage in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]]
    )

# === Robust ZIP Export Function ===
def create_zip_export(image, confidences, plot_fig, segmentations, lesion_names, recommendation):
    """Create a ZIP file with all analysis results"""
    print("Starting ZIP export...")
    try:
        if not image:
            print("No image available for export")
            return None
            
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "dr_analysis_results.zip")
        print(f"Creating ZIP at: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Save original image
            img_path = os.path.join(temp_dir, "original.png")
            image.save(img_path)
            zipf.write(img_path, "original.png")
            print("Added original image to ZIP")
            
            # Save classification results
            csv_path = os.path.join(temp_dir, "classification.csv")
            pd.DataFrame({
                "DR Stage": list(confidences.keys()),
                "Confidence": list(confidences.values())
            }).to_csv(csv_path, index=False)
            zipf.write(csv_path, "classification.csv")
            print("Added classification CSV to ZIP")
            
            # Save confidence plot
            plot_path = os.path.join(temp_dir, "confidence.png")
            plot_fig.savefig(plot_path, bbox_inches='tight')
            zipf.write(plot_path, "confidence.png")
            print("Added confidence plot to ZIP")
            
            # Save segmentation images
            for i, name in enumerate(lesion_names):
                seg_path = os.path.join(temp_dir, f"{name}.png")
                seg_img = Image.fromarray(segmentations[i])
                seg_img.save(seg_path)
                zipf.write(seg_path, f"{name}_segmentation.png")
            print("Added segmentation images to ZIP")
            
            # Save recommendation
            rec_path = os.path.join(temp_dir, "recommendation.txt")
            with open(rec_path, "w", encoding="utf-8") as f:
                f.write(recommendation)
            zipf.write(rec_path, "clinical_recommendation.txt")
            print("Added recommendation to ZIP")
            
            # Add a README
            readme_path = os.path.join(temp_dir, "README.txt")
            with open(readme_path, "w") as f:
                f.write("Diabetic Retinopathy Analysis Results\n")
                f.write("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            zipf.write(readme_path, "README.txt")
            print("Added README to ZIP")
        
        print(f"ZIP export successful: {zip_path}")
        return zip_path
    except Exception as e:
        print(f"❌ ZIP export failed: {str(e)}")
        print(traceback.format_exc())
        return None
        
    # === Enhanced PDF Creation with Unicode Support ===
class UnicodePDF(FPDF):
    """PDF class with UTF-8 support for emojis and special characters"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add DejaVu font (requires font files)
        self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        self.set_font('DejaVu', '', 12)
    
    def header(self):
        # Custom header
        self.set_font('DejaVu', 'B', 15)
        self.cell(0, 10, 'Diabetic Retinopathy Analysis Report', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        # Custom footer
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def clean_text_for_pdf(text):
    """Replace emojis with text representations for PDF compatibility"""
    replacements = {
        "🧪": "[TEST]",
        "💊": "[MED]",
        "📅": "[CALENDAR]",
        "🥗": "[FOOD]",
        "⚠️": "[WARNING]"
    }
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    return text

# === Enhanced PDF Creation with Unicode Support ===
def create_pdf_report(image, confidences, plot_fig, segmentations, lesion_names, recommendation):
    """Create a PDF report with all analysis results"""
    print("Starting PDF report generation...")
    try:
        if not image:
            print("No image available for report")
            return None
            
        temp_dir = tempfile.mkdtemp()
        pdf = FPDF()
        pdf.add_page()
        print("Created PDF page")
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Diabetic Retinopathy Analysis Report", 0, 1, "C")
        pdf.ln(10)
        
        # Original image
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Original Retinal Image:", 0, 1)
        img_path = os.path.join(temp_dir, "original.png")
        image.save(img_path)
        pdf.image(img_path, x=10, w=190)
        pdf.ln(10)
        print("Added original image to PDF")
        
        # Classification results
        pdf.cell(0, 10, "DR Classification Results:", 0, 1)
        pdf.ln(5)
        
        # Add table
        col_width = pdf.w / 3
        pdf.set_font("Arial", "B", 12)
        pdf.cell(col_width, 10, "DR Stage", border=1)
        pdf.cell(col_width, 10, "Confidence", border=1)
        pdf.cell(col_width, 10, "Severity", border=1, ln=1)
        pdf.set_font("Arial", size=10)
        
        for stage, conf in confidences.items():
            pdf.cell(col_width, 10, stage, border=1)
            pdf.cell(col_width, 10, f"{conf:.4f}", border=1)
            pdf.cell(col_width, 10, stage, border=1, ln=1)
        
        pdf.ln(10)
        print("Added classification table to PDF")
        
        # Confidence plot
        plot_path = os.path.join(temp_dir, "confidence.png")
        plot_fig.savefig(plot_path, bbox_inches='tight')
        pdf.cell(0, 10, "Confidence Distribution:", 0, 1)
        pdf.image(plot_path, x=10, w=190)
        pdf.ln(10)
        print("Added confidence plot to PDF")
        
        # Segmentation results
        pdf.cell(0, 10, "Lesion Segmentation Results:", 0, 1)
        pdf.ln(5)
        
        for i, name in enumerate(lesion_names):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"{name.capitalize()} Segmentation:", 0, 1)
            
            seg_path = os.path.join(temp_dir, f"{name}.png")
            seg_img = Image.fromarray(segmentations[i])
            seg_img.save(seg_path)
            
            pdf.image(seg_path, x=10, w=190)
            pdf.ln(5)
        print("Added segmentation images to PDF")
        
        # Clinical recommendation
        pdf.cell(0, 10, "Clinical Recommendations:", 0, 1)
        pdf.set_font("Arial", size=11)
        clean_rec = clean_text_for_pdf(recommendation)
        pdf.multi_cell(0, 8, clean_rec)
        print("Added recommendations to PDF")
        
        # Save PDF
        pdf_path = os.path.join(temp_dir, "dr_report.pdf")
        pdf.output(pdf_path)
        print(f"PDF report generated: {pdf_path}")
        
        return pdf_path
    except Exception as e:
        print(f"❌ PDF generation failed: {str(e)}")
        print(traceback.format_exc())
        return None
# Global variable to store latest results
latest_results = {
    "image": None,
    "confidences": None,
    "plot_fig": None,
    "segmentations": None,
    "recommendation": None
}

# Add these functions outside the Gradio interface block
def toggle_section(visible):
    """Toggle section visibility"""
    return not visible, gr.update(visible=not visible)

# Update the metric card creation to use consistent naming
def create_metric_card(metric_name, value):
    """Create HTML for a metric card with dynamic value"""
    # Convert value to percentage for progress bar
    width = value * 100
    
    # Map to consistent display names
    display_names = {
        "Jaccard": "IoU",
        "F1 Score": "Dice",
        "Precision": "Precision",
        "Recall": "Recall"
    }
    display_name = display_names.get(metric_name, metric_name)
    
    return f"""
    <div class='metric-card'>
        <div class='metric-label'>{display_name}</div>
        <div class='metric-value'>{value:.4f}</div>
        <div class='progress-container'>
            <div class='progress-bar' style='width: {width}%'></div>
        </div>
    </div>
    """

def update_stage_cards(no_dr, mild, moderate, severe, proliferative):
    """Update all stage cards with new confidence values"""
    cards = []
    for stage, value in zip(
        ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        [no_dr, mild, moderate, severe, proliferative]
    ):
        # Convert to percentage width for the bar
        width = value * 100
        cards.append(f"""
        <div class='metric-card'>
            <div class='metric-label'>{stage}</div>
            <div class='metric-value'>{value:.4f}</div>
            <div class='severity-indicator'>
                <div class='severity-bar' style='width: {width}%; background: {label_colors[stage]};'></div>
            </div>
        </div>
        """)
    return cards

# === Enhanced CSS for Modern UI ===
modern_css = """
:root {
    --primary: #4a6491;
    --secondary: #8bc34a;
    --accent: #2196f3;
    --danger: #f44336;
    --success: #4caf50;
    --warning: #ff9800;
    --light: #f8f9fa;
    --dark: #2c3e50;
    --border-radius: 16px;
    --shadow: 0 8px 30px rgba(0,0,0,0.12);
    --transition: all 0.3s ease;
}

body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
    min-height: 100vh;
    margin: 0;
    padding: 20px;
    color: #333;
}

#container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 20px;
}

.header {
    text-align: center;
    padding: 40px 0 20px;
    margin-bottom: 30px;
}

.header h1 {
    font-size: 2.8rem;
    margin-bottom: 15px;
    color: #ffffff; /* Set text color to white */
    font-weight: 800;
}
.header h2 {
    font-size: 2.2rem;
    margin-bottom: 15px;
    color: #ffffff; /* Set text color to white */
    font-weight: 800;
}

.header p {
    font-size: 1.2rem;
    max-width: 800px;
    margin: 0 auto 25px;
    color: #4a5568;
    line-height: 1.6;
}

.card {
    background: white;
    border-radius: var(--border-radius);
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: var(--shadow);
    border: none;
    transition: var(--transition);
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
}

.card-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 25px;
    color: var(--dark);
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 15px;
    border-bottom: 2px solid #f0f4f8;
}

.card-title i {
    font-size: 1.8rem;
    color: var(--accent);
}

.tab-buttons {
    display: flex;
    gap: 15px;
    margin-bottom: 25px;
    justify-content: center;
}

.tab-btn {
    background: white !important;
    border: 2px solid var(--primary) !important;
    color: var(--primary) !important;
    font-weight: 600 !important;
    padding: 12px 25px !important;
    border-radius: 50px !important;
    transition: var(--transition) !important;
    font-size: 1.1rem !important;
}

.tab-btn:hover, .tab-btn:focus, .tab-btn.active {
    background: var(--primary) !important;
    color: white !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 7px 20px rgba(74, 100, 145, 0.3) !important;
}

.upload-area {
    border: 3px dashed #cbd5e0;
    border-radius: var(--border-radius);
    padding: 40px 20px;
    text-align: center;
    background: #f8fafc;
    transition: var(--transition);
    margin-bottom: 25px;
    cursor: pointer;
}

.upload-area:hover {
    border-color: var(--accent);
    background: #f0f7ff;
}

.example-label {
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 15px;
    font-weight: 600;
    color: var(--dark);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.metric-card {
    background: linear-gradient(to bottom right, #f7fafc, #ebf4ff);
    padding: 20px 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    border-top: 4px solid var(--accent);
    transition: var(--transition);
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--dark);
    margin: 15px 0;
    font-family: 'Segoe UI', sans-serif;
}

.metric-label {
    font-size: 1rem;
    font-weight: 600;
    color: #4a5568;
}

.severity-indicator {
    height: 10px;
    border-radius: 5px;
    background: #e2e8f0;
    margin-top: 12px;
    overflow: hidden;
}

.severity-bar {
    height: 100%;
    border-radius: 5px;
}

.download-btn {
    background: linear-gradient(135deg, var(--success) 0%, #2ecc71 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 14px 30px !important;
    border-radius: 50px !important;
    box-shadow: 0 7px 20px rgba(46, 204, 113, 0.3) !important;
    transition: var(--transition) !important;
    margin-top: 15px;
    font-size: 1.1rem !important;
}

.download-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 25px rgba(46, 204, 113, 0.4) !important;
}

.primary-btn {
    background: linear-gradient(135deg, var(--primary) 0%, #4a6491 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 16px 40px !important;
    border-radius: 50px !important;
    font-size: 1.2rem !important;
    box-shadow: 0 7px 20px rgba(74, 100, 145, 0.3) !important;
    transition: var(--transition) !important;
    margin: 20px 0;
}

.primary-btn:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 12px 30px rgba(74, 100, 145, 0.4) !important;
}

.segmentation-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 25px;
    margin-top: 20px;
}

.segmentation-card {
    background: white;
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    transition: var(--transition);
}

.segmentation-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
}

.segmentation-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--dark) 100%);
    color: white;
    padding: 20px;
    font-weight: 700;
    font-size: 1.2rem;
    display: flex;
    align-items: center;
    gap: 12px;
}

.segmentation-content {
    padding: 20px;
    text-align: center;
}

.recommendation-box {
    background: linear-gradient(to bottom right, #f0f7ff, #e3f2fd);
    border-radius: var(--border-radius);
    padding: 30px;
    margin-top: 30px;
    border-left: 5px solid var(--accent);
    font-family: 'Segoe UI', sans-serif;
    line-height: 1.7;
    font-size: 1.1rem;
}

.recommendation-box h3 {
    color: var(--primary);
    margin-top: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 15px;
    border-bottom: 2px solid #cfe8fc;
    font-size: 1.5rem;
}

.recommendation-content {
    background: black;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.05);
    font-size: 1.1rem;
}

.status-badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    margin-left: 15px;
}

.status-0 { background: #4CAF50; color: white; }
.status-1 { background: #8BC34A; color: white; }
.status-2 { background: #FFC107; color: #333; }
.status-3 { background: #FF9800; color: white; }
.status-4 { background: #F44336; color: white; }

.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.85);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    flex-direction: column;
    gap: 25px;
    backdrop-filter: blur(5px);
}

.spinner {
    width: 60px;
    height: 60px;
    border: 6px solid rgba(74, 100, 145, 0.2);
    border-top: 6px solid var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 30px;
    font-size: 1rem;
    color: #718096;
    border-top: 1px solid #e2e8f0;
    font-weight: 500;
}

.comparison-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
    color: black !important;
}

.comparison-item {
    text-align: center;
    background: white;
    color: black; 
    padding: 20px;
    border-radius: var(--border-radius);
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    transition: var(--transition);

    color: black !important;
}

.comparison-item:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

.performance-highlight {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-radius: var(--border-radius);
    padding: 25px;
    margin: 30px 0;
    border-left: 5px solid var(--accent);
    color: black !important;
}

.toggle-section {
    cursor: pointer;
    padding: 15px 20px;
    background: #f1f8fe;
    border-radius: var(--border-radius);
    margin: 20px 0;
    font-weight: 700;
    display: flex;
    align-items: center;
    color: var(--dark);
    font-size: 1.1rem;
    transition: var(--transition);
}

.toggle-section:hover {
    background: #e3f2fd;
}

.toggle-section::before {
    content: "►";
    margin-right: 15px;
    font-size: 0.9rem;
    transition: transform 0.3s;
}

.toggle-section.active::before {
    transform: rotate(90deg);
}
"""

# Add this function to create a loading spinner
def create_loading_spinner():
    return """
    <div class="loading-overlay">
        <div class="spinner"></div>
        <h3>Analyzing Retinal Image</h3>
        <p>This may take a few moments...</p>
    </div>
    """

# Define lesion names and keys for the sub-tabs
lesion_types = [
    ("Blood Vessels", "vessels"),
    ("Hemorrhages", "hemorrhages"),
    ("Microaneurysms", "microaneurysms"),
    ("Hard Exudates", "exudates"),
    ("Optic Disc", "optic_disc"),
    ("Soft Exudates", "macula")
]

# === Toggle section JavaScript ===
toggle_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('.toggle-section').forEach(section => {
        section.addEventListener('click', () => {
            section.classList.toggle('active');
            const content = section.nextElementSibling;
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
        });
    });
});
</script>
"""

# === Loading spinner JavaScript ===
loading_js = """
<script>
function showLoading() {
    document.querySelector('.loading-overlay').style.display = 'flex';
}
function hideLoading() {
    document.querySelector('.loading-overlay').style.display = 'none';
}
</script>
"""

with gr.Blocks(css=modern_css, theme=gr.themes.Soft()) as dr_app:
    # Add JavaScript
    gr.HTML(toggle_js)
    gr.HTML(loading_js)
    
    # Create a component for the loading overlay
    loading_overlay = gr.HTML(visible=False)
    

    
    # Add loading overlay (hidden by default)
    gr.HTML(f"""
    <div class="loading-overlay" style="display: none;">
        <div class="spinner"></div>
        <h3>Analyzing Retinal Image</h3>
        <p>This may take a few moments...</p>
    </div>
    """)
    
    with gr.Column(elem_id="container"):
        # Header
        with gr.Column(elem_classes="header"):
            gr.Markdown("""
            <h1>👁️ EyeNOVA</h1>          
            <h1>Advanced Diabetic Retinopathy Analyzer</h1>
            <p>AI-powered retinal image analysis for early detection of diabetic eye complications</p>
            """)
        
        # Tab navigation
        with gr.Row(elem_classes="tab-buttons"):
            tab1_btn = gr.Button("DR Detection & Segmentation", elem_classes="tab-btn active")
            tab2_btn = gr.Button("Individual Lesion Analysis", elem_classes="tab-btn")
            tab3_btn = gr.Button("Research Tools", elem_classes="tab-btn")
        
        # Main content area
        with gr.Column():
            # === TAB 1: DR Detection & Segmentation ===
            with gr.Column(visible=True) as tab1_content:
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes="card"):
                            gr.Markdown("""<div class="card-title"><i>📤</i> Upload Retinal Image</div>""")
                            
                            # Upload area
                            with gr.Column(elem_classes="upload-area"):
                                image_input = gr.Image(type="pil", label="", show_label=False, interactive=True)
                                gr.Markdown("Drag & drop an image or click to browse")
                            
                            # Examples
                            gr.Markdown("**Try sample images:**", elem_classes="example-label")
                            gr.Examples(
                                examples=[
                                    "data/sample/10_left.jpeg",
                                    "data/sample/10_right.jpeg",
                                    "data/sample/15_left.jpeg",
                                    "data/sample/16_right.jpeg",
                                ],
                                inputs=image_input,
                                label="",
                                examples_per_page=4
                            )
                            
                            # Action button
                            analyze_btn = gr.Button("Analyze Image", variant="primary", elem_classes="primary-btn")
                            
                            # Export buttons
                            with gr.Row():
                                export_btn = gr.Button("Export as ZIP", elem_classes="download-btn")
                                pdf_btn = gr.Button("Generate PDF Report", elem_classes="download-btn")
                            export_output = gr.File(label="Download Export", visible=False)
                            report_output = gr.File(label="Download Report", visible=False)
                    
                    # Right column - Results
                    with gr.Column(scale=2):
                        # DR Classification Results
                        with gr.Column(elem_classes="card", visible=False) as results_card:
                            gr.Markdown("""<div class="card-title"><i>📊</i> DR Classification Results</div>""")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    output_label = gr.Label(label="Prediction Confidence", show_label=False)
                                with gr.Column(scale=2):
                                    plot_output = gr.Plot()
                            
                            # Enhanced metric cards - now dynamic
                            with gr.Column(elem_classes="metric-grid"):
                                # Create HTML components for each stage
                                stage_cards = []
                                for stage in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]:
                                    # Create initial HTML with 0.00 value
                                    html = gr.HTML(f"""
                                    <div class='metric-card'>
                                        <div class='metric-label'>{stage}</div>
                                        <div class='metric-value'>0.00</div>
                                        <div class='severity-indicator'>
                                            <div class='severity-bar' style='width: 0%; background: {label_colors[stage]};'></div>
                                        </div>
                                    </div>
                                    """)
                                    stage_cards.append(html)
                        
                        # Segmentation Results
                        with gr.Column(elem_classes="card", visible=False) as seg_card:
                            gr.Markdown("""<div class="card-title"><i>🔍</i> Lesion Segmentation</div>""")
                            
                            with gr.Column(elem_classes="segmentation-grid"):
                                seg_titles = [
                                    ("Blood Vessels", "vessels"),
                                    ("Hemorrhages", "hemorrhages"),
                                    ("Microaneurysms", "microaneurysms"),
                                    ("Hard Exudates", "exudates"),
                                    ("Optic Disc", "optic_disc"),
                                    ("Soft Exudates", "macula")
                                ]
                                
                                seg_outputs = []
                                for title, key in seg_titles:
                                    with gr.Column(elem_classes="segmentation-card"):
                                        gr.Markdown(f"""<div class="segmentation-header"><i>🔬</i> {title}</div>""")
                                        with gr.Column(elem_classes="segmentation-content"):
                                            seg_img = gr.Image(label="", show_label=False, interactive=False)
                                            seg_outputs.append(seg_img)
                        
                        # Recommendations
                        with gr.Column(elem_classes="card", visible=False) as rec_card:
                            gr.Markdown("""<div class="card-title"><i>🩺</i> Clinical Recommendations</div>""")
                            with gr.Column(elem_classes="recommendation-box"):
                                gr.Markdown("""<h3><i>📝</i> Professional Guidance</h3>""")
                                with gr.Column(elem_classes="recommendation-content"):
                                    recommendation_output = gr.Markdown("")
            
            # === TAB 2: Individual Lesion Analysis ===
            with gr.Column(visible=False) as tab2_content:
                # Create a tab group for the 6 lesion types
                for name, key in lesion_types:
                    with gr.Tab(f"{name}", elem_classes="sub-tab"):
                        gr.Markdown(f"## 🔍 {name} Analysis")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 📁 Upload Ground Truth Mask")
                                gt_input = gr.Image(type="numpy", label=f"{name} Mask", image_mode='L')
                            with gr.Column():
                                submit_btn = gr.Button(f"Analyze {name}", variant="primary", elem_classes="primary-btn")
                        
                        # Results area
                        with gr.Column(elem_classes="results-box"):
                            gr.Markdown("### 🔍 Segmentation Results")
                            
                            # === Section 3: Comparison Grid ===
                            with gr.Column(elem_classes="comparison-grid"):
                                # Create separate components for each image type
                                with gr.Row():
                                    # Original Image
                                    with gr.Column(elem_classes="comparison-item"):
                                        gr.Markdown("**Original**")
                                        original_img = gr.Image(interactive=False, height=200)
                                    # Ground Truth
                                    with gr.Column(elem_classes="comparison-item"):
                                        gr.Markdown("**Ground Truth**")
                                        gt_img = gr.Image(interactive=False, height=200)
                                    # Prediction
                                    with gr.Column(elem_classes="comparison-item"):
                                        gr.Markdown("**Prediction**")
                                        pred_img = gr.Image(interactive=False, height=200)
                            
                            # === ADD STATE VARIABLES HERE ===
                            detailed_metrics_visible = gr.State(False)
                            vis_options_visible = gr.State(False)
                                    
                            
                            # Performance Highlight
                            with gr.Column(elem_classes="performance-highlight"):
                                gr.Markdown("### Performance Summary")
                                with gr.Row(elem_classes="metric-grid"):
                                    # Create HTML components for each metric
                                    jaccard_card = gr.HTML(create_metric_card("Jaccard", 0.0))
                                    f1_card = gr.HTML(create_metric_card("F1 Score", 0.0))
                                    precision_card = gr.HTML(create_metric_card("Precision", 0.0))
                                    recall_card = gr.HTML(create_metric_card("Recall", 0.0))
                            
                            # Toggle Sections
                             # === REPLACE THE OLD TOGGLE SECTIONS WITH THIS ===
                            with gr.Row():
                                detailed_toggle = gr.Button("Detailed Metrics", elem_classes="toggle-section")
                                
                            
                            with gr.Column(visible=False) as detailed_metrics:
                                metrics_label = gr.Label(label="Segmentation Metrics", elem_classes="metric-box")
                                

                            
                            # === ADD EVENT HANDLING FOR TOGGLES HERE ===
                            detailed_toggle.click(
                                toggle_section,
                                inputs=[detailed_metrics_visible],
                                outputs=[detailed_metrics_visible, detailed_metrics]
                            )
                            
                            
                            
                            gr.Markdown("### 📈 Analysis Visualizations")
                            with gr.Row(elem_classes="plot-row"):
                                with gr.Column(elem_classes="plot-container"):
                                    metrics_plot = gr.Plot(label="Metrics Comparison")
                                with gr.Column(elem_classes="plot-container"):
                                    confusion_plot = gr.Plot(label="Confusion Matrix")
                            
                            with gr.Row(elem_classes="plot-row"):
                                pixel_plot = gr.Plot(label="Pixel Distribution")
                        
                        
        
                        # Event handling
                        submit_btn.click(
                                            fn=lambda img, gt, lesion_key=key: analyze_single_lesion(img, gt, lesion_key),
                                            inputs=[image_input, gt_input],
                                            outputs=[
                                                original_img,    # Original image
                                                gt_img,          # Ground truth
                                                pred_img,        # Prediction
                                                metrics_label,
                                                metrics_plot,
                                                confusion_plot,
                                                pixel_plot,
                                                jaccard_card,    # Jaccard/IoU metric card
                                                f1_card,         # F1/Dice metric card
                                                precision_card,  # Precision metric card
                                                recall_card 
                                            ]
                                        )
            
            # === TAB 3: Research Tools ===
            with gr.Column(visible=False) as tab3_content:
                with gr.Column(elem_classes="card"):
                    gr.Markdown("""<div class="card-title"><i>🔬</i> Research Tools</div>""")
                    
                    gr.Markdown("### Model Comparison")
                    model_dropdown = gr.Dropdown(["UNet", "DeepLabV3", "Attention UNet"], 
                                                label="Model Architecture", 
                                                value="UNet")
                    
                    gr.Markdown("### Quantitative Analysis")
                    metrics_df = gr.Dataframe(
                        headers=["Lesion", "IoU", "Dice", "Precision", "Recall"],
                        value=[["Vessels", 0.92, 0.95, 0.94, 0.91]],
                        interactive=True
                    )
                    
                    gr.Markdown("### Export Options")
                    with gr.Row():
                        gr.Button("Export Metrics as CSV", variant="secondary", elem_classes="download-btn")
                        gr.Button("Save Visualizations", variant="secondary", elem_classes="download-btn")
        
        # Footer
        gr.Markdown(
            "**Note**: This AI tool provides preliminary screening and should not replace professional medical diagnosis. Always consult an ophthalmologist for comprehensive eye care.",
            elem_classes="footer"
        )
    
    # Tab navigation event handling
    def set_active_tab(tab_name):
        return [
            gr.update(visible=tab_name == "tab1"),
            gr.update(visible=tab_name == "tab2"),
            gr.update(visible=tab_name == "tab3"),
            gr.update(variant="primary" if tab_name == "tab1" else "secondary"),
            gr.update(variant="primary" if tab_name == "tab2" else "secondary"),
            gr.update(variant="primary" if tab_name == "tab3" else "secondary")
        ]
        
    tab1_btn.click(lambda: set_active_tab("tab1"), None, 
                [tab1_content, tab2_content, tab3_content, tab1_btn, tab2_btn, tab3_btn])
    tab2_btn.click(lambda: set_active_tab("tab2"), None, 
                [tab1_content, tab2_content, tab3_content, tab1_btn, tab2_btn, tab3_btn])
    tab3_btn.click(lambda: set_active_tab("tab3"), None, 
                [tab1_content, tab2_content, tab3_content, tab1_btn, tab2_btn, tab3_btn])
    # Event handling for analysis
    analyze_btn.click(
                fn=lambda: gr.HTML(create_loading_spinner(), visible=True),
                inputs=None,
                outputs=[loading_overlay],
                queue=False
            ).then(
                fn=run_analysis,
                inputs=image_input,
                outputs=[
                    output_label, 
                    plot_output,
                    *seg_outputs,
                    recommendation_output,
                    *stage_cards   
                ]
            ).then(
                fn=lambda: [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)],
                inputs=None,
                outputs=[results_card, seg_card, rec_card],
                queue=False
            ).then(
                fn=update_stage_cards,
                inputs=stage_cards,  # inputs are the confidence values
                outputs=stage_cards,  # outputs are updated HTML
                queue=False
            ).then(
                fn=lambda: gr.HTML(visible=False),
                inputs=None,
                outputs=[loading_overlay],
                queue=False
            )

    # Event handling for exports
    export_btn.click(
        fn=lambda: create_zip_export(
            latest_results["image"],
            latest_results["confidences"],
            latest_results["plot_fig"],
            latest_results["segmentations"],
            list(model_sizes.keys()),
            latest_results["recommendation"]
        ) if latest_results.get("image") else None,
        inputs=[],
        outputs=export_output,
        api_name="export_zip"
    ).then(
        fn=lambda path: gr.update(visible=path is not None),
        inputs=[export_output],
        outputs=export_output,
        queue=False
    )

    pdf_btn.click(
        fn=lambda: create_pdf_report(
            latest_results["image"],
            latest_results["confidences"],
            latest_results["plot_fig"],
            latest_results["segmentations"],
            list(model_sizes.keys()),
            latest_results["recommendation"]
        ) if latest_results.get("image") else None,
        inputs=[],
        outputs=report_output,
        api_name="export_pdf"
    ).then(
        fn=lambda path: gr.update(visible=path is not None),
        inputs=[report_output],
        outputs=report_output,
        queue=False
    )

def normalize_mask(mask):
    """Universal mask normalization for any input format"""
    # Convert to numpy array if needed
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Handle different dimensionalities
    if len(mask.shape) == 3:
        # Handle multi-channel masks:
        # 1. Check if it's RGB with identical channels (common in PNG exports)
        if mask.shape[2] == 3 and np.array_equal(mask[:,:,0], mask[:,:,1]) and np.array_equal(mask[:,:,0], mask[:,:,2]):
            mask = mask[:,:,0]  # Use first channel
        # 2. Check if it's RGBA
        elif mask.shape[2] == 4:
            mask = mask[:,:,0]  # Use first channel (often the meaningful one)
        # 3. Otherwise use the maximum across channels
        else:
            mask = np.max(mask, axis=2)
    
    # Normalize value ranges
    if mask.dtype == np.uint16:
        mask = mask.astype(np.float32) / 65535 * 255
    elif mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = mask * 255
    elif np.max(mask) <= 1.0:
        mask = mask * 255
    
    # Convert to uint8
    mask = mask.astype(np.uint8)
    
    # Binarize - consider anything above 10% as positive
    _, binary_mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    
    return binary_mask

def calculate_metrics(pred, gt):
    """Robust metric calculation with comprehensive error handling"""
    # Handle null cases
    if gt is None:
        return {"Jaccard": 0.0, "F1 Score": 0.0, "Recall": 0.0, "Precision": 0.0, "Accuracy": 0.0}
    
    # Ensure compatible sizes
    if gt.shape != pred.shape:
        try:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"Resize error: {str(e)}")
            return {"Jaccard": 0.0, "F1 Score": 0.0, "Recall": 0.0, "Precision": 0.0, "Accuracy": 0.0}
    
    # Binarize masks
    try:
        pred_bin = (pred > 25).astype(np.uint8).flatten()  # More tolerant threshold
        gt_bin = (gt > 25).astype(np.uint8).flatten()
    except Exception as e:
        print(f"Binarization error: {str(e)}")
        return {"Jaccard": 0.0, "F1 Score": 0.0, "Recall": 0.0, "Precision": 0.0, "Accuracy": 0.0}
    
    # Handle special cases
    total_pixels = gt_bin.size
    if total_pixels == 0:
        return {"Jaccard": 0.0, "F1 Score": 0.0, "Recall": 0.0, "Precision": 0.0, "Accuracy": 0.0}
    
    if np.sum(gt_bin) == 0:  # No positive pixels in GT
        if np.sum(pred_bin) == 0:  # Perfect match - no lesions
            return {"Jaccard": 1.0, "F1 Score": 1.0, "Recall": 1.0, "Precision": 1.0, "Accuracy": 1.0}
        else:  # False positives
            return {"Jaccard": 0.0, "F1 Score": 0.0, "Recall": 0.0, "Precision": 0.0, "Accuracy": 0.0}
    
    # Calculate metrics
    try:
        
        return {
        "Jaccard": round(jaccard_score(gt_bin, pred_bin), 4),
        "F1 Score": round(f1_score(gt_bin, pred_bin), 4),
        "Recall": round(recall_score(gt_bin, pred_bin), 4),
        "Precision": round(precision_score(gt_bin, pred_bin), 4),
         "Accuracy": round(accuracy_score(gt_bin, pred_bin), 4),
     }
    except Exception as e:
        print(f"Metric calculation error: {str(e)}")
        return {"Jaccard": 0.0, "F1 Score": 0.0, "Recall": 0.0, "Precision": 0.0, "Accuracy": 0.0}
# def calculate_metrics(pred, gt):
#     if gt is None or np.sum(gt) == 0:
#         return {"Jaccard": 0.0, "F1 Score": 0.0, "Recall": 0.0, "Precision": 0.0, "Accuracy": 0.0}
    
#     pred_bin = (pred > 127).flatten()
#     gt_bin = (gt > 127).flatten()
    
#     return {
#         "Jaccard": round(jaccard_score(gt_bin, pred_bin), 4),
#         "F1 Score": round(f1_score(gt_bin, pred_bin), 4),
#         "Recall": round(recall_score(gt_bin, pred_bin), 4),
#         "Precision": round(precision_score(gt_bin, pred_bin), 4),
#         "Accuracy": round(accuracy_score(gt_bin, pred_bin), 4),
#     }
# === Enhanced function for individual lesion analysis ===
def analyze_single_lesion(image, gt_mask, lesion_key):
    """
    Analyze a single lesion type with ground truth comparison
    """
    # Get the segmentation model
    model_seg = seg_models.get(lesion_key)
    size = model_sizes.get(lesion_key, (512, 512))
    
    # Run segmentation
    if model_seg is None:
        # Create placeholder images
        placeholder = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(placeholder, "Model Not Loaded", (50, 256), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return placeholder, placeholder, placeholder, "", None, None, None
    
    # Convert PIL image to numpy array (RGB)
    original_np = np.array(image)
    
    # Resize original to model size
    original_display = cv2.resize(original_np, size)
    
    # Process ground truth mask
    if gt_mask is not None:
        gt_mask = normalize_mask(gt_mask)
        gt_display = cv2.resize(gt_mask, size, interpolation=cv2.INTER_NEAREST)
        if len(gt_display.shape) == 2:
            gt_display = cv2.cvtColor(gt_display, cv2.COLOR_GRAY2RGB)
    else:
        gt_mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        gt_display = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Run prediction
    pred_mask = predict_segmentation(image, model_seg, size)
    
    # Process prediction mask
    pred_mask_normalized = normalize_mask(pred_mask)
    if len(pred_mask_normalized.shape) == 3 and pred_mask_normalized.shape[2] == 3:
        pred_gray = cv2.cvtColor(pred_mask_normalized, cv2.COLOR_RGB2GRAY)
    else:
        pred_gray = pred_mask_normalized
    
    # Calculate metrics
    metrics = calculate_metrics(pred_gray, gt_mask)
    formatted_metrics = {k: f"{v:.4f}" for k, v in metrics.items()}
    metrics_str = "\n".join([f"{k}: {v}" for k, v in formatted_metrics.items()])

    
    # Create analysis visualizations
    metrics_fig = plot_metrics(metrics)
    confusion_fig = plot_confusion_matrix(gt_mask, pred_gray)
    pixel_fig = plot_pixel_distribution(gt_mask, pred_gray)
    
    return (
        original_display, 
        gt_display, 
        pred_mask, 
        metrics_str, 
        metrics_fig, 
        confusion_fig, 
        pixel_fig,
        metrics["Jaccard"],  # IoU
        metrics["F1 Score"], # Dice
        metrics["Precision"],
        metrics["Recall"]
    )


def plot_metrics(metrics):
    """Create bar chart for segmentation metrics"""
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color='#4C78A8')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score')
    ax.set_title('Segmentation Metrics')
    ax.invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    plt.tight_layout()
    return fig



def plot_confusion_matrix(gt, pred):
    """Create confusion matrix visualization with size normalization"""
    # Ensure both masks are the same size
    if gt.shape != pred.shape:
        # Resize ground truth to match prediction size
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Flatten arrays
    gt_flat = gt.flatten() / 255
    pred_flat = pred.flatten() / 255
    
    # Calculate confusion matrix
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    
    # Create matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", 
                    ha="center", va="center", 
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=14)
    
    # Labels and title
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
    ax.set_yticklabels(['Actual Negative', 'Actual Positive'])
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig

def plot_pixel_distribution(gt, pred):
    """Create pixel distribution comparison plot with size normalization"""
    # Ensure both masks are the same size
    if gt.shape != pred.shape:
        # Resize ground truth to match prediction size
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Flatten and normalize
    gt_flat = (gt.flatten() / 255).astype(int)
    pred_flat = (pred.flatten() / 255).astype(int)
    
    # Calculate distributions
    gt_counts = [np.sum(gt_flat == 0), np.sum(gt_flat == 1)]
    pred_counts = [np.sum(pred_flat == 0), np.sum(pred_flat == 1)]
    
    # Plot
    bar_width = 0.35
    index = np.arange(2)
    
    ax.bar(index, gt_counts, bar_width, label='Ground Truth', color='#54A24B')
    ax.bar(index + bar_width, pred_counts, bar_width, label='Prediction', color='#4C78A8')
    
    ax.set_xlabel('Pixel Class')
    ax.set_ylabel('Count')
    ax.set_title('Pixel Class Distribution')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Background', 'Lesion'])
    ax.legend()
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    dr_app.launch()
