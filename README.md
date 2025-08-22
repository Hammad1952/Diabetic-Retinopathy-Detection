🩺 **EyeNOVA: Multi-Lesion Retinal Segmentation & Diabetic Retinopathy Detection**

EyeNOVA is a deep learning–powered diagnostic platform developed as part of my undergraduate thesis. It combines retinal lesion segmentation and disease classification to provide accurate, interpretable, and scalable solutions for Diabetic Retinopathy (DR) screening.

This project demonstrates expertise in Machine Learning, Computer Vision, AI integration, and Full-Stack Development, reflecting both research-level innovation and production-ready engineering.

🎯 **Project Highlights**

**Lesion Segmentation**

Designed and trained 6 U-Net models to segment retinal structures: vessels, exudates, hemorrhages, microaneurysms, and optic disc.

Applied PyTorch Lightning for reproducible experiments and scalable training.

**Hybrid Classification**

Integrated DenseNet-169 (CNN) with Vision Transformer (ViT) for DR severity grading.

Achieved Quadratic Weighted Kappa (QWK) = 0.925 and AUC > 0.90 on benchmark datasets.

**Clinical Decision Support**

Connected with Gemini Pro API to generate automated, stage-specific treatment recommendations.

Designed interpretable outputs with heatmaps and overlays for better clinical trust.

**Interactive Deployment**

Built a Gradio-based user interface for real-time predictions and visualization.

Backend services powered by FastAPI, supporting modular scaling and deployment.



**⚙️ Installation**
git clone https://github.com/Hammad1952/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

🚀**Usage**

Run the Gradio application:

python app.py


You can:

Upload a retinal fundus image

Get segmentation maps for lesions and retinal structures

Receive classification results (DR stage)

View clinical recommendations generated automatically

**📊 Results**

**Segmentation Performance:** High Dice and Jaccard scores across lesion types.

**Classification Accuracy:** QWK = 0.925, outperforming several baseline models.

**Explainability:** Model interpretability enhanced via Grad-CAM and lesion overlays.

**📦 Datasets**

The system was trained and validated using publicly available datasets:

APTOS 2019

MESSIDOR

EyePACS

📌 Datasets are not included due to size and licensing. Preprocessing scripts and links are provided.

**💾 Pretrained Models**

Due to file size limits, pretrained weights are hosted externally:



**🛠️ Tech Stack**

Languages & Frameworks

Python (PyTorch, TensorFlow, PyTorch Lightning, FastAPI, Gradio)

Frontend: HTML, CSS, JavaScript (for dashboards & reports)

[Gradio UI Screenshot]
(https://github.com/Hammad1952/Diabetic-Retinopathy-Detection/blob/master/screencapture-127-0-0-1-7860-2025-08-17-15_31_53.png)

**👨‍💻 Author**

Hammad Ali Khan

🎓 BS Computer System Engineering, Mirpur University of Science & Technology (MUST)

📧 hammadkhanjadoon61@gmail.com

🌐 Portfolio : https://hammad-ali-khan-jadoon.vercel.app/
