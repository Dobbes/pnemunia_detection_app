import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import random

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
        
        try:
            # Try DenseNet-121 which often performs well on medical images
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
        except Exception as e2:
            print(f"Error loading DenseNet model: {e2}")
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
    """Create advanced data augmentation transforms for better training"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),  # Simulate image noise
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def apply_histogram_equalization(image):
    """Apply histogram equalization to improve image contrast"""
    image_np = np.array(image)
    
    # Convert to YCrCb color space for better equalization
    if len(image_np.shape) == 3:  # Color image
        # Apply equalization only to the luminance channel
        import cv2
        ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(equalized)
    else:  # Grayscale image
        equalized = cv2.equalizeHist(image_np)
        return Image.fromarray(equalized)
    
    return image  # Original image if transformation fails

def fine_tune_model(model, processor, training_images, num_epochs=8, class_weights=None):
    """
    Fine-tune the model on the selected pneumonia images
    
    In a real application, this would be more extensive with proper training data.
    Enhanced with data augmentation, learning rate scheduling, and more epochs.
    """
    # Make the model ready for training
    model.train()
    
    # Enhanced with lower learning rate for more stable fine-tuning
    learning_rate = 5e-6  # Reduced from 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Add learning rate scheduler to decrease LR when progress plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Create data augmentation transforms
    augmentation = create_data_augmentation()
    
    # Set up loss function with class weights if provided
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    # Separate normal and pneumonia images for balanced training
    normal_images = [img for img in training_images if 'normal' in img.lower()]
    pneumonia_images = [img for img in training_images if 'pneumonia' in img.lower() or ('person' in img.lower() and 'bacteria' in img.lower()) or ('person' in img.lower() and 'virus' in img.lower())]
    
    # Calculate class weights based on the number of samples if not provided
    if class_weights is None and len(normal_images) > 0 and len(pneumonia_images) > 0:
        normal_weight = len(training_images) / (2 * len(normal_images))
        pneumonia_weight = len(training_images) / (2 * len(pneumonia_images))
        class_weights = [normal_weight, pneumonia_weight]
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    print(f"Fine-tuning model for {num_epochs} epochs with {len(training_images)} images")
    print(f"Normal images: {len(normal_images)}, Pneumonia images: {len(pneumonia_images)}")
    
    # Track best loss for early stopping
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # For this demo, we'll just simulate a few update steps
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Shuffle image order for each epoch
        random.shuffle(training_images)
        
        # Simulate batch training with all available images
        for img_path in training_images:
            try:
                # Prepare the image
                image = Image.open(img_path).convert('RGB')
                
                # Apply preprocessing and augmentation
                image = apply_histogram_equalization(image)
                
                # Convert PIL image to tensor and apply augmentation
                augmented_tensor = augmentation(image).unsqueeze(0)  # Add batch dimension
                
                # Determine label based on filename
                is_pneumonia = 'pneumonia' in img_path.lower() or ('person' in img_path.lower() and ('bacteria' in img_path.lower() or 'virus' in img_path.lower()))
                label = torch.tensor([1 if is_pneumonia else 0])
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(pixel_values=augmented_tensor)
                loss = loss_fn(outputs.logits, label)
                
                # Calculate accuracy for monitoring
                _, predicted = torch.max(outputs.logits, 1)
                correct_predictions += (predicted == label).sum().item()
                total_predictions += label.size(0)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(training_images)
        epoch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Update learning rate based on loss
        scheduler.step(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
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
        normal_files = [f for f in os.listdir(normal_dir)[:30] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in normal_files:
            img_path = os.path.join(normal_dir, img_file)
            try:
                # Apply histogram equalization for better contrast
                image = Image.open(img_path).convert('RGB')
                processed_image = apply_histogram_equalization(image)
                
                # Create inputs using the processor
                inputs = processor(images=processed_image, return_tensors="pt")
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
            except Exception as e:
                print(f"Error processing normal image {img_file}: {e}")
    
    # Process pneumonia images
    pneumonia_dir = os.path.join(dataset_path, 'PNEUMONIA')
    if os.path.exists(pneumonia_dir):
        pneumonia_files = [f for f in os.listdir(pneumonia_dir)[:30] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in pneumonia_files:
            img_path = os.path.join(pneumonia_dir, img_file)
            try:
                # Apply histogram equalization for better contrast
                image = Image.open(img_path).convert('RGB')
                processed_image = apply_histogram_equalization(image)
                
                # Create inputs using the processor
                inputs = processor(images=processed_image, return_tensors="pt")
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
            except Exception as e:
                print(f"Error processing pneumonia image {img_file}: {e}")
    
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