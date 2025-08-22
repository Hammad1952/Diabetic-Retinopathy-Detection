import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from src.model import DRModel
import lightning as L
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, 
    MulticlassRecall, MulticlassF1Score, 
    MulticlassCohenKappa, MulticlassConfusionMatrix,
    MulticlassAUROC
)
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class FolderDataset(Dataset):
    """Dataset for loading images from folder structure"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Map class names to numerical labels
        self.class_names = ['0', '1', '2', '3', '4']
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # Collect image paths and labels
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = read_image(image_path)
        except Exception as e:
            raise IOError(f"Error loading image at path '{image_path}': {e}")
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                raise RuntimeError(f"Error applying transformations to image at path '{image_path}': {e}")
        return image, label

def main(args):
    # Load model from checkpoint
    
    model = DRModel.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    model.freeze()

    # Test transformations
    test_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Add conversion to RGB if needed:
        # T.Lambda(lambda x: x[:3])  # Keep only first 3 channels if there's alpha
    ])

    # Load test dataset from folder structure
    test_dataset = FolderDataset(args.test_folder, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize metrics
    num_classes = model.num_classes
    metrics = {
        "accuracy": MulticlassAccuracy(num_classes=num_classes),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "kappa": MulticlassCohenKappa(num_classes=num_classes, weights='quadratic'),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=num_classes),
        "auc": MulticlassAUROC(num_classes=num_classes, average="macro")
    }

    # Move metrics to device
    for metric in metrics.values():
        metric.to(device)

    # Collect predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

    # Concatenate results
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # Calculate metrics
    results = {}
    for name, metric in metrics.items():
        if name == "confusion_matrix":
            results[name] = metric(all_preds, all_labels)
        else:
            results[name] = metric(all_probs, all_labels) if name == "auc" else metric(all_preds, all_labels)
    
    # Print metrics
    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy: {results['accuracy'].item():.4f}")
    print(f"Precision (Macro): {results['precision'].item():.4f}")
    print(f"Recall (Macro): {results['recall'].item():.4f}")
    print(f"F1 Score (Macro): {results['f1'].item():.4f}")
    print(f"AUC: {results['auc'].item():.4f}")
    print(f"Cohen's Kappa: {results['kappa'].item():.4f}")
    
    # Define class names for diabetic retinopathy
    class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative DR (4)']
    
    # Classification report
    print("\n===== Classification Report =====")
    print(classification_report(
        all_labels.cpu().numpy(),
        all_preds.cpu().numpy(),
        target_names=class_names
    ))
    
    # Confusion matrix visualization
    print("\n===== Confusion Matrix =====")
    cm = results["confusion_matrix"].cpu().numpy()
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix as confusion_matrix.png")
    
    # Save results to CSV
    if args.output_csv:
        report = classification_report(
            all_labels.cpu().numpy(),
            all_preds.cpu().numpy(),
            target_names=class_names,
            output_dict=True
        )
        df_report = pd.DataFrame(report).transpose()
        
        # Add overall metrics
        metrics_df = pd.DataFrame({
            'Accuracy': [results['accuracy'].item()],
            'Precision': [results['precision'].item()],
            'Recall': [results['recall'].item()],
            'F1': [results['f1'].item()],
            'AUC': [results['auc'].item()],
            'Kappa': [results['kappa'].item()]
        })
        df_report = pd.concat([metrics_df, df_report])
        
        df_report.to_csv(args.output_csv)
        print(f"\nSaved results to {args.output_csv}")

    # Save raw predictions for error analysis
    predictions_df = pd.DataFrame({
        'image_path': test_dataset.image_paths,
        'true_label': all_labels.cpu().numpy(),
        'predicted_label': all_preds.cpu().numpy()
    })
    for i in range(num_classes):
        predictions_df[f'prob_class_{i}'] = all_probs[:, i].cpu().numpy()
    predictions_df.to_csv('raw_predictions.csv', index=False)
    print("Saved raw predictions to raw_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DR classification model")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--test_folder", type=str, required=True, 
                        help="Path to test folder with class subdirectories")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loader workers")
    parser.add_argument("--image_size", type=int, default=224, 
                        help="Image size (must match training)")
    parser.add_argument("--output_csv", type=str, default="test_results.csv", 
                        help="Output CSV file path")
    
    args = parser.parse_args()
    main(args)