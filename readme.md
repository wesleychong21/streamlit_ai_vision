# Image Object Detection


This is a Streamlit web application that performs object detection using a pre-trained Faster R-CNN model on images. The model can detect 80 different object categories from the COCO dataset.

## Features

- Upload images in JPG, JPEG, or PNG format
- Real-time object detection
- Adjustable confidence threshold
- Display of detected objects with bounding boxes
- Individual cropped views of detected objects
- Confidence scores for each detection

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
git clone https://github.com/wesleychong21/streamlit_ai_vision.git

2. Navigate to the project directory:
cd streamlit_ai_vision

3. Install the required dependencies:
pip install -r requirements.txt

## How to use the application

1. Run the application by running `streamlit run app.py`
2. The application will open in your default web browser. If it doesn't, navigate to `http://localhost:8501`

3. Use the application:
   - Click on "Choose an image..." to upload an image
   - Click the "Detect Objects" button to run object detection
   - Adjust the confidence threshold slider to filter detections
   - View the results, including:
     - Original image
     - Image with detection boxes
     - Individual cropped objects with confidence scores

## Model Information

The application uses the Faster R-CNN ResNet-50 FPN V2 model pre-trained on the COCO dataset, which can detect 80 different object categories including:
- People
- Vehicles (cars, buses, bikes)
- Animals (dogs, cats, birds)
- Common objects (chairs, bottles, phones)
- And many more

## Limitations

- The model works best with clear, well-lit images
- Processing time may vary depending on image size and hardware capabilities
- The model is limited to detecting the 80 categories defined in the COCO dataset

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

[Add your license information here]