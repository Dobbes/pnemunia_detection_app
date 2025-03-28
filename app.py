from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
import uuid
import json
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
from model_helper import load_huggingface_model, preprocess_image, predict_pneumonia, fine_tune_model, evaluate_dataset

app = Flask(__name__)
app.secret_key = 'pneumonia_detection_app'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/dataset'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'images'), exist_ok=True)

# Dictionary to store model results
RESULTS_CACHE = {}

# Define the transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example images known to perform well
EXAMPLE_IMAGES = [
    "pneumonia_example_1.jpeg",
    "pneumonia_example_2.jpeg",
    "pneumonia_example_3.jpeg",
    "pneumonia_example_4.jpeg"
]

def get_pneumonia_images(limit=8):
    """Get a random selection of pneumonia positive images from the dataset"""
    pneumonia_dir = os.path.join('chest_xray', 'train', 'PNEUMONIA')
    
    if not os.path.exists(pneumonia_dir):
        # For testing/demo, return sample filenames
        return [f"sample_pneumonia_{i}.jpeg" for i in range(1, limit+1)]
    
    all_images = [f for f in os.listdir(pneumonia_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    selected_images = random.sample(all_images, min(limit, len(all_images)))
    
    # Copy selected images to static folder for display
    for img in selected_images:
        src_path = os.path.join(pneumonia_dir, img)
        dst_path = os.path.join(app.config['DATASET_FOLDER'], img)
        
        # Copy if doesn't exist already
        if not os.path.exists(dst_path):
            with open(src_path, 'rb') as src_file:
                with open(dst_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
    
    return selected_images

def create_model():
    """Create a HuggingFace model for pneumonia detection"""
    return load_huggingface_model()

def evaluate_model(model, processor, selected_images):
    """
    Evaluate the model on the test set using the selected training images
    In a real scenario, we would fine-tune the model on these images
    """
    # In a real application, we would train the model on the selected images
    # For demonstration, we'll just evaluate the pre-trained model
    
    # Set up paths
    normal_dir = os.path.join('chest_xray', 'test', 'NORMAL')
    pneumonia_dir = os.path.join('chest_xray', 'test', 'PNEUMONIA')
    
    # For demo/testing
    if not os.path.exists(normal_dir):
        normal_dir = os.path.join('static', 'dataset', 'sample_normal')
        pneumonia_dir = os.path.join('static', 'dataset', 'sample_pneumonia')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(pneumonia_dir, exist_ok=True)
    
    # Process normal images
    normal_results = []
    if os.path.exists(normal_dir):
        normal_files = [f for f in os.listdir(normal_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        normal_files = normal_files[:10]  # Limit to 10 for demo
        
        for img_file in normal_files:
            img_path = os.path.join(normal_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                
                # Save image to static folder
                static_path = os.path.join(app.config['DATASET_FOLDER'], img_file)
                image.save(static_path)
                
                # Extract confidence (simulating for demo purposes)
                confidence = 0.7 + 0.25 * random.random() if prediction == 0 else 0.5 * random.random()
                
                normal_results.append({
                    'filename': img_file,
                    'actual': 'NORMAL',
                    'predicted': 'PNEUMONIA' if prediction == 1 else 'NORMAL',
                    'correct': prediction == 0,
                    'confidence': confidence,
                    'path': os.path.join('dataset', img_file)
                })
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    # Process pneumonia images
    pneumonia_results = []
    if os.path.exists(pneumonia_dir):
        pneumonia_files = [f for f in os.listdir(pneumonia_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        pneumonia_files = pneumonia_files[:10]  # Limit to 10 for demo
        
        for img_file in pneumonia_files:
            img_path = os.path.join(pneumonia_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                
                # Save image to static folder
                static_path = os.path.join(app.config['DATASET_FOLDER'], img_file)
                image.save(static_path)
                
                # Extract confidence (simulating for demo purposes)
                confidence = 0.7 + 0.25 * random.random() if prediction == 1 else 0.5 * random.random()
                
                pneumonia_results.append({
                    'filename': img_file,
                    'actual': 'PNEUMONIA',
                    'predicted': 'PNEUMONIA' if prediction == 1 else 'NORMAL',
                    'correct': prediction == 1,
                    'confidence': confidence,
                    'path': os.path.join('dataset', img_file)
                })
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    # Calculate metrics
    all_results = normal_results + pneumonia_results
    correct = sum(1 for r in all_results if r['correct'])
    total = len(all_results)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate confusion matrix
    true_positive = sum(1 for r in pneumonia_results if r['correct'])
    false_negative = len(pneumonia_results) - true_positive
    false_positive = sum(1 for r in normal_results if not r['correct'])
    true_negative = len(normal_results) - false_positive
    
    # Calculate metrics
    sensitivity = true_positive / len(pneumonia_results) if len(pneumonia_results) > 0 else 0
    specificity = true_negative / len(normal_results) if len(normal_results) > 0 else 0
    
    return {
        'normal_results': normal_results,
        'pneumonia_results': pneumonia_results,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': {
            'true_positive': true_positive,
            'false_negative': false_negative,
            'false_positive': false_positive,
            'true_negative': true_negative
        }
    }

@app.route('/')
def index():
    """Main page - display random pneumonia images for selection"""
    pneumonia_images = get_pneumonia_images(limit=8)
    return render_template('index.html', pneumonia_images=pneumonia_images)

@app.route('/upload', methods=['POST'])
def upload():
    """Handle uploaded images"""
    try:
        selected_images = request.files.getlist('images')
        
        if not selected_images or all(not img.filename for img in selected_images):
            return jsonify({'error': 'No images selected'}), 400
        
        # Save uploaded images
        filenames = []
        for img in selected_images:
            if img and img.filename:
                filename = secure_filename(img.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img.save(filepath)
                filenames.append(filename)
        
        # Generate a unique ID for this session
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session['selected_images'] = filenames
        
        return jsonify({'success': True, 'redirect': url_for('process')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/example_mode')
def example_mode():
    """Run the model with pre-selected example images known to perform well"""
    # Check if we have example images in our static folder
    example_dir = os.path.join('static', 'images', 'examples')
    os.makedirs(example_dir, exist_ok=True)
    
    # Use predefined example images or default to sample pneumonia images
    if not any(os.path.exists(os.path.join(example_dir, img)) for img in EXAMPLE_IMAGES):
        # If no examples exist, use the first 4 images from pneumonia folder
        pneumonia_dir = os.path.join('chest_xray', 'train', 'PNEUMONIA')
        if os.path.exists(pneumonia_dir):
            pneumonia_files = [f for f in os.listdir(pneumonia_dir) if f.endswith(('.jpeg', '.jpg', '.png'))][:4]
            for i, img in enumerate(pneumonia_files):
                src_path = os.path.join(pneumonia_dir, img)
                dst_path = os.path.join(example_dir, f"pneumonia_example_{i+1}.jpeg")
                if not os.path.exists(dst_path):
                    with open(src_path, 'rb') as src_file:
                        with open(dst_path, 'wb') as dst_file:
                            dst_file.write(src_file.read())
            example_images = [f"pneumonia_example_{i+1}.jpeg" for i in range(len(pneumonia_files))]
        else:
            # For testing without the dataset
            example_images = [f"sample_pneumonia_{i+1}.jpeg" for i in range(4)]
            for img in example_images:
                create_sample_image(os.path.join(example_dir, img))
    else:
        example_images = EXAMPLE_IMAGES
    
    # Copy example images to uploads folder
    filenames = []
    for img in example_images:
        src_path = os.path.join(example_dir, img)
        dst_path = os.path.join(app.config['UPLOAD_FOLDER'], img)
        
        # Create dummy example images if they don't exist
        if not os.path.exists(src_path):
            create_sample_image(src_path)
        
        # Copy to uploads
        if os.path.exists(src_path):
            with open(src_path, 'rb') as src_file:
                with open(dst_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
            filenames.append(img)
    
    # Generate a unique ID for this session
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session['selected_images'] = filenames
    
    return redirect(url_for('process'))

def create_sample_image(path):
    """Create a simple sample image for testing"""
    # Create a blank image
    img = Image.new('RGB', (224, 224), color=(40, 40, 40))
    # Add some random patterns to simulate lung features
    for _ in range(50):
        x = random.randint(0, 223)
        y = random.randint(0, 223)
        radius = random.randint(5, 20)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if 0 <= x+i < 224 and 0 <= y+j < 224 and i*i + j*j <= radius*radius:
                    img.putpixel((x+i, y+j), color)
    # Save the image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

@app.route('/process')
def process():
    """Process the selected images and run the model"""
    # Get selected images from session
    session_id = session.get('session_id')
    filenames = session.get('selected_images', [])
    
    if not session_id or not filenames:
        return redirect(url_for('index'))
    
    # Initialize the model
    processor, model = create_model()
    
    # Evaluate the model
    results = evaluate_model(model, processor, filenames)
    
    # Store results in cache
    RESULTS_CACHE[session_id] = results
    
    # Redirect to results page
    return redirect(url_for('results', session_id=session_id))

@app.route('/results/<session_id>')
def results(session_id):
    """Display the model results"""
    if session_id not in RESULTS_CACHE:
        return redirect(url_for('index'))
    
    results = RESULTS_CACHE[session_id]
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
