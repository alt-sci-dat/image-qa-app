from tempfile import NamedTemporaryFile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel, Field
from ultralytics import YOLO
import streamlit as st
from tools import ImageCaptionTool
import shutil
import re
import easyocr
import cv2
import numpy as np

# Initialize tools
caption_tool = ImageCaptionTool()
detection_model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
reader = easyocr.Reader(['en'])  # Initialize EasyOCR for English text

def detect_text_in_image(image_path):
    """Detect and read text from the image"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return [], None
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get text detection results
        results = reader.readtext(image)
        
        # Filter results with confidence > 0.5
        valid_results = [(bbox, text, prob) for bbox, text, prob in results if prob > 0.5]
        
        return valid_results, image.shape
    except Exception as e:
        st.error(f"Error in text detection: {str(e)}")
        return [], None

def analyze_question(question, caption, object_counts, image_path):
    """Analyze the question and generate appropriate response"""
    question = question.lower().strip()
    
    # Question categories and their keywords
    categories = {
        'text': ['text', 'written', 'says', 'say', 'words', 'word', 'shirt', 't-shirt', 'clothing', 'read', 'lettering'],
        'color': ['color', 'colour', 'colored', 'coloured', 'hue', 'tone', 'shade'],
        'position': ['where', 'position', 'location', 'place', 'situated', 'standing', 'sitting', 'lying'],
        'action': ['doing', 'action', 'activity', 'performing', 'engaged', 'involved'],
        'emotion': ['emotion', 'feeling', 'mood', 'expression', 'happy', 'sad', 'angry', 'smiling', 'frowning'],
        'time': ['time', 'when', 'period', 'day', 'night', 'morning', 'evening', 'afternoon'],
        'weather': ['weather', 'sunny', 'rainy', 'cloudy', 'stormy', 'clear', 'foggy'],
        'count': ['how many', 'number of', 'count', 'quantity', 'amount'],
        'description': ['what', 'describe', 'tell me about', 'looks like', 'appears'],
        'attribute': ['wearing', 'holding', 'carrying', 'using', 'has', 'have', 'with'],
        'comparison': ['bigger', 'smaller', 'larger', 'taller', 'shorter', 'more', 'less', 'than'],
        'identity': ['who', 'person', 'people', 'man', 'woman', 'child', 'boy', 'girl']
    }
    
    # Determine question category
    question_category = None
    for category, keywords in categories.items():
        if any(keyword in question for keyword in keywords):
            question_category = category
            break
    
    # Handle text-related questions
    if question_category == 'text':
        text_results, image_shape = detect_text_in_image(image_path)
        if text_results and image_shape is not None:
            if any(word in question for word in ['shirt', 't-shirt', 'clothing']):
                clothing_text = []
                for bbox, text, prob in text_results:
                    y_coord = bbox[0][1]
                    if y_coord < image_shape[0] * 0.6:
                        clothing_text.append(text)
                if clothing_text:
                    return f"The text on the clothing appears to be: {', '.join(clothing_text)}"
                return "I cannot clearly read any text on the clothing in the image."
            else:
                all_text = [text for _, text, _ in text_results]
                return f"The text in the image reads: {', '.join(all_text)}"
        return "No text was detected in the image."
    
    # Handle color-related questions
    elif question_category == 'color':
        color_words = [word for word in caption.split() if word.lower() in ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'pink', 'brown', 'gray', 'grey']]
        if color_words:
            return f"The image contains these colors: {', '.join(color_words)}."
        return "I cannot determine specific colors in the image."
    
    # Handle position-related questions
    elif question_category == 'position':
        location_words = [word for word in caption.split() if word.lower() in ['inside', 'outside', 'room', 'house', 'building', 'street', 'park', 'beach', 'forest', 'water', 'rock', 'near', 'beside', 'next to', 'in front of', 'behind']]
        if location_words:
            return f"The scene appears to be {', '.join(location_words)}."
        return "I cannot determine the specific position or location from the image."
    
    # Handle action-related questions
    elif question_category == 'action':
        action_words = [word for word in caption.split() if word.lower() in ['standing', 'sitting', 'walking', 'running', 'jumping', 'holding', 'carrying', 'looking', 'watching', 'reading', 'writing', 'talking', 'smiling', 'laughing']]
        if action_words:
            return f"The person appears to be {', '.join(action_words)}."
        return "I cannot determine the specific action from the image."
    
    # Handle emotion-related questions
    elif question_category == 'emotion':
        emotion_words = [word for word in caption.split() if word.lower() in ['happy', 'sad', 'angry', 'smiling', 'laughing', 'frowning', 'crying', 'excited', 'surprised', 'scared']]
        if emotion_words:
            return f"The person appears to be {', '.join(emotion_words)}."
        return "I cannot determine the specific emotion from the image."
    
    # Handle time-related questions
    elif question_category == 'time':
        time_words = [word for word in caption.split() if word.lower() in ['morning', 'afternoon', 'evening', 'night', 'dawn', 'dusk', 'sunny', 'dark']]
        if time_words:
            return f"The image appears to be taken during {', '.join(time_words)}."
        return "I cannot determine the time of day from the image."
    
    # Handle weather-related questions
    elif question_category == 'weather':
        weather_words = [word for word in caption.split() if word.lower() in ['sunny', 'rainy', 'cloudy', 'stormy', 'clear', 'foggy', 'windy']]
        if weather_words:
            return f"The weather appears to be {', '.join(weather_words)}."
        return "I cannot determine the weather conditions from the image."
    
    # Handle counting questions
    elif question_category == 'count':
        for obj in object_counts:
            if obj in question:
                return f"There {'is' if object_counts[obj] == 1 else 'are'} {object_counts[obj]} {obj}{'s' if object_counts[obj] != 1 else ''} in the image."
        return "Please specify what objects you want to count."
    
    # Handle description questions
    elif question_category == 'description':
        return f"Based on the image: {caption}"
    
    # Handle attribute questions
    elif question_category == 'attribute':
        attribute_words = [word for word in caption.split() if word.lower() in ['wearing', 'holding', 'carrying', 'using', 'has', 'have', 'with']]
        if attribute_words:
            return f"The person is {', '.join(attribute_words)}."
        return "I cannot determine specific attributes from the image."
    
    # Handle comparison questions
    elif question_category == 'comparison':
        return "I cannot make comparisons between objects in the image. Please ask about specific objects or attributes."
    
    # Handle identity questions
    elif question_category == 'identity':
        if 'person' in object_counts:
            return f"There {'is' if object_counts['person'] == 1 else 'are'} {object_counts['person']} person{'s' if object_counts['person'] != 1 else ''} in the image."
        return "I cannot detect any people in the image."
    
    # Default response for unrecognized questions
    return "I can answer questions about:\n" + \
           "- Text in the image\n" + \
           "- Colors and visual elements\n" + \
           "- Position and location\n" + \
           "- Actions and activities\n" + \
           "- Emotions and expressions\n" + \
           "- Time of day\n" + \
           "- Weather conditions\n" + \
           "- Object counts\n" + \
           "- General description\n" + \
           "- Attributes and characteristics\n" + \
           "- People and their identities"

# set title
st.title('Ask a question to an image')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question about your image:')

    ##############################
    ### process image and question ###
    ##############################
    # Create a temporary file with proper extension
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, file.name)
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())

    try:
        # write response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                # Get image caption
                caption = caption_tool._run(temp_path)
                
                # Get object detections using YOLOv8
                results = detection_model(temp_path)
                detections = results[0].boxes.data.tolist()
                class_names = results[0].names
                
                # Count objects
                object_counts = {}
                for det in detections:
                    class_id = int(det[5])
                    class_name = class_names[class_id]
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                # Display results
                st.write("Image Description:", caption)
                st.write("\nDetected Objects:")
                for obj, count in object_counts.items():
                    st.write(f"- {obj}: {count}")
                
                # Analyze question and generate response
                response = analyze_question(user_question, caption, object_counts, temp_path)
                st.write("\nAnswer:", response)
                
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
