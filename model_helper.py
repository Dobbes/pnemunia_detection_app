import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim

def load_huggingface_model():
    """
    Load a pre-trained HuggingFace model for pneumonia detection
    
    Using a Vision Transformer (ViT) model which works better for medical images
    """
    try:
        # Use a Vision Transformer model pre-trained on ImageNet which transfers well to medical images
        model_name = "google/vit-base-patch16-224"
        
        # Initialize the processor and model
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "NORMAL", 1: "PNEUMONIA"},
            label2id={"NORMAL": 0, "PNEUMONIA": 1},
            ignore_mismatched_sizes=True
        )
        
        # Return the processor and model
        return processor, model
    except Exception as e:
        # If ViT model fails, fallback to ResNet-50
        print(f"Error loading ViT model: {e}")
        print("Falling back to ResNet-50")
        
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        
        # For binary classification, we need to handle the model differently
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=2,
            id2label={0: "NORMAL", 1: "PNEUMONIA"},
            label2id={"NORMAL": 0, "PNEUMONIA": 1},
            ignore_mismatched_sizes=True
        )
        
        return processor, model

def preprocess_image(image_path, processor):
    """Preprocess an image for the model"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def predict_pneumonia(model, preprocessed_inputs):
    """Use the model to predict pneumonia"""
    with torch.no_grad():
        outputs = model(**preprocessed_inputs)
    
    # Get the predicted class (0: Normal, 1: Pneumonia)
    logits = outputs.logits
    predictions = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = predictions[0][predicted_class].item()
    
    return {
        'class': 'PNEUMONIA' if predicted_class == 1 else 'NORMAL',
        'confidence': confidence,
        'is_pneumonia': predicted_class == 1,
        'logits': logits.tolist(),  # For debugging
        'class_probs': {
            'NORMAL': predictions[0][0].item(),
            'PNEUMONIA': predictions[0][1].item()
        }
    }

def create_data_augmentation():
    """Create data augmentation transforms for better training"""
    from torchvision import transforms
    
    # Define augmentation transformations to make the model more robust
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def fine_tune_model(model, processor, training_images, num_epochs=3):
    """
    Fine-tune the model on the selected pneumonia images
    
    In a real application, this would be more extensive with proper training data.
    For this demo, we'll simulate fine-tuning.
    """
    # Make the model ready for training
    model.train()
    
    # Set up optimizer with a low learning rate to avoid overfitting
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # In a real scenario, we would:
    # 1. Create a proper dataset with both pneumonia and normal samples
    # 2. Use a DataLoader for batch processing
    # 3. Set up a validation set for early stopping
    # 4. Train for multiple epochs with learning rate scheduling
    
    # For this demo, we'll just simulate a few update steps
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Simulate a single batch of data
        for img_path in training_images:
            # Prepare the image
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            
            # Assume all selected images are pneumonia for this example
            labels = torch.tensor([1])  # 1 for pneumonia
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(training_images)}")
    
    # Set the model back to evaluation mode
    model.eval()
    return model

def evaluate_dataset(model, processor, dataset_path):
    """
    Evaluate the model on a dataset of images
    
    Args:
        model: The HuggingFace model
        processor: The image processor
        dataset_path: Path to the dataset folder with NORMAL and PNEUMONIA subfolders
    
    Returns:
        Dictionary of evaluation results with metrics
    """
    results = {
        'normal_results': [],
        'pneumonia_results': [],
        'confusion_matrix': {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0
        }
    }
    
    # Process normal images
    normal_dir = os.path.join(dataset_path, 'NORMAL')
    if os.path.exists(normal_dir):
        normal_files = [f for f in os.listdir(normal_dir)[:20] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in normal_files:
            img_path = os.path.join(normal_dir, img_file)
            inputs = preprocess_image(img_path, processor)
            prediction = predict_pneumonia(model, inputs)
            
            is_correct = not prediction['is_pneumonia']
            
            if is_correct:
                results['confusion_matrix']['true_negative'] += 1
            else:
                results['confusion_matrix']['false_positive'] += 1
                
            results['normal_results'].append({
                'filename': img_file,
                'actual': 'NORMAL',
                'predicted': prediction['class'],
                'confidence': prediction['confidence'],
                'correct': is_correct,
                'path': os.path.join('dataset', os.path.basename(img_path))
            })
    
    # Process pneumonia images
    pneumonia_dir = os.path.join(dataset_path, 'PNEUMONIA')
    if os.path.exists(pneumonia_dir):
        pneumonia_files = [f for f in os.listdir(pneumonia_dir)[:20] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in pneumonia_files:
            img_path = os.path.join(pneumonia_dir, img_file)
            inputs = preprocess_image(img_path, processor)
            prediction = predict_pneumonia(model, inputs)
            
            is_correct = prediction['is_pneumonia']
            
            if is_correct:
                results['confusion_matrix']['true_positive'] += 1
            else:
                results['confusion_matrix']['false_negative'] += 1
                
            results['pneumonia_results'].append({
                'filename': img_file,
                'actual': 'PNEUMONIA',
                'predicted': prediction['class'],
                'confidence': prediction['confidence'],
                'correct': is_correct,
                'path': os.path.join('dataset', os.path.basename(img_path))
            })
    
    # Calculate metrics
    tp = results['confusion_matrix']['true_positive']
    tn = results['confusion_matrix']['true_negative']
    fp = results['confusion_matrix']['false_positive']
    fn = results['confusion_matrix']['false_negative']
    
    total = tp + tn + fp + fn
    correct = tp + tn
    
    # Standard metrics
    results['accuracy'] = correct / total if total > 0 else 0
    results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Additional metrics
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['f1_score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    return results

def apply_gradcam(model, processor, image_path, target_layer=None):
    """
    Apply Grad-CAM visualization technique to highlight important regions in the image
    
    This is useful to understand what parts of the X-ray the model is focusing on
    
    Args:
        model: The HuggingFace model
        processor: The image processor
        image_path: Path to the image
        target_layer: Target layer for Grad-CAM (if None, uses the last layer)
        
    Returns:
        Heatmap visualization as a PIL Image
    """
    # This would require pytorch-grad-cam package
    # For demo purposes, we'll return a simulated heatmap
    
    try:
        # Open the original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Create a simple simulated heatmap (this would be replaced with actual Grad-CAM)
        heatmap = Image.new('RGB', original_image.size, (0, 0, 0))
        
        # In a real implementation, we would:
        # 1. Use pytorch-grad-cam to compute the actual heatmap
        # 2. Overlay it on the original image
        # 3. Return the overlaid image
        
        return heatmap
    except Exception as e:
        print(f"Error applying Grad-CAM: {e}")
        return None
