# app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
from PIL import Image
import osa
import tempfile
import base64
import requests
from io import BytesIO

app = Flask(__name__)

def preprocess_captcha(image):
    """Process an image loaded as a numpy array"""
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply thresholding to create a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove noise
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Dilate to connect broken parts of characters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    return dilated

def enhance_recognition_with_multiple_methods(image):
    """Attempt multiple preprocessing methods to improve accuracy"""
    # Original method
    processed1 = preprocess_captcha(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Alternative method with different thresholding
    _, processed2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Another alternative with Otsu's thresholding
    _, processed3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply different tesseract configs
    configs = [
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ]
    
    results = []
    
    # Try each processing method with each config
    for processed in [processed1, processed2, processed3]:
        pil_img = Image.fromarray(processed)
        for config in configs:
            text = pytesseract.image_to_string(pil_img, config=config)
            clean_text = ''.join(c for c in text if c.isalnum())
            if clean_text:
                results.append(clean_text)
    
    # Manual confirmation based on visual inspection
    expected_length = 6  # Typical CAPTCHA length, adjust as needed
    
    # Filter results by expected length
    valid_results = [r for r in results if len(r) == expected_length]
    
    if valid_results and len(valid_results) > 0:
        # Find the most common result
        from collections import Counter
        most_common = Counter(valid_results).most_common(1)[0][0]
        return most_common
    elif results:
        # If no result matches expected length, return the most common
        return Counter(results).most_common(1)[0][0]
    else:
        return ""

@app.route('/api/recognize', methods=['POST'])
def recognize_captcha():
    try:
        data = request.get_json()
        
        # Check if image is provided either as URL or base64
        if 'image_url' in data:
            # Download image from URL
            response = requests.get(data['image_url'])
            if response.status_code != 200:
                return jsonify({"error": "Failed to download image from URL"}), 400
            
            # Convert to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
        elif 'image_base64' in data:
            # Decode base64 image
            image_data = base64.b64decode(data['image_base64'])
            image_array = np.asarray(bytearray(BytesIO(image_data).read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
        else:
            return jsonify({"error": "No image provided. Please provide 'image_url' or 'image_base64'"}), 400
            
        # Process the image
        result = enhance_recognition_with_multiple_methods(image)
        final_text = result.replace(" ", "").strip()
        
        return jsonify({
            "success": True,
            "captcha_text": final_text
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "CAPTCHA Recognition API",
        "endpoints": {
            "/api/recognize": "POST - Recognize CAPTCHA from image URL or base64"
        },
        "example": {
            "image_url": "https://example.com/captcha.jpg"
        }
    })

# For local development
if __name__ == '_main_':
    app.run(debug=True)