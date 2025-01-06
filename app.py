import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import transforms
from PIL import Image
import json
import numpy as np
import cv2

# Load COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@st.cache_resource
def load_model():
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model.eval()
    return model

def detect_objects(image, model):
    # Transform image
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    
    # Get predictions
    with torch.no_grad():
        prediction = model([img_tensor])
    
    return prediction[0]

def draw_boxes(image, prediction, confidence_threshold=0.5):
    img_np = np.array(image)
    original_img_np = img_np.copy()  # Make a copy for cropping
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    
    # Store cropped images and their details
    detections = []
    
    for box, score, label in zip(boxes, scores, labels):
        if score > confidence_threshold:
            x1, y1, x2, y2 = box.astype(int)
            # Draw rectangle on the main image
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            label_text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
            cv2.putText(img_np, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Crop from the original image without boxes
            crop = Image.fromarray(original_img_np[y1:y2, x1:x2])
            detections.append({
                'crop': crop,
                'label': COCO_INSTANCE_CATEGORY_NAMES[label],
                'score': score
            })
    
    return img_np, detections

# Streamlit app
st.title("Object Detection with COCO Dataset")

# Load model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Detect objects
    if st.button('Detect Objects'):
        prediction = detect_objects(image, model)
        
        # Draw boxes on image
        confidence_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5)
        result_image, detections = draw_boxes(image, prediction, confidence_threshold)
        
        # Display result
        st.image(result_image, caption='Detection Result', use_container_width=True)
        
        # Display detection details with cropped images
        st.write("Detection Details:")
        for i, detection in enumerate(detections):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(detection['crop'], width=100)
            with col2:
                st.write(f"Object {i+1}: {detection['label']} (Confidence: {detection['score']:.2f})")
