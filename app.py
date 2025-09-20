import os
import base64
import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template

# Make sure you have the GOOGLE_API_KEY environment variable set or uncomment the line below
# and replace 'YOUR_API_KEY' with your actual API key.
# It is best practice to set this as an environment variable to keep it secure.
# genai.configure(api_key="YOUR_API_KEY") 
genai.configure()

app = Flask(__name__)

# Define the directory for temporary file storage
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# --- OMR Processing Logic (Computer Vision) ---

# Predefined answer key for a 10-question OMR sheet.
# This would need to be updated to match the real answer key for your test.
# The keys are question numbers (1-10) and the values are the correct answers (A-D).
ANSWER_KEY = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'A',
    6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'B'
}

def process_omr_sheet(image_path):
    """
    Processes an OMR sheet image to extract answers and score it.
    
    Args:
        image_path (str): The path to the uploaded image file.
        
    Returns:
        A tuple containing:
        - score (int): The total number of correct answers.
        - answers (dict): A dictionary of the user's answers.
    """
    
    # Read the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur the image to reduce noise and threshold it to get a binary image.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # We will assume a fixed OMR sheet layout for simplicity.
    # The circles are a fixed size and in a grid. We need to identify them.
    
    # A list to store the detected answer bubbles
    bubble_contours = []
    
    # Find all contours that are roughly circle-shaped (based on aspect ratio and area)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        
        # Filter for contours that are likely to be bubbles
        if 0.9 <= aspect_ratio <= 1.1 and area > 100:
            bubble_contours.append(c)
            
    # Sort contours from top to bottom
    bubble_contours = sorted(bubble_contours, key=cv2.contourArea, reverse=True)
    bubble_contours = sorted(bubble_contours, key=lambda c: cv2.boundingRect(c)[1])
    
    # Process the top 100 circles to find the answers (assuming a 10-question test with 4 options each)
    detected_answers = {}
    
    for i in range(10):  # Loop for each question from 0 to 9
        question_bubbles = bubble_contours[i*4 : (i+1)*4]
        
        # Sort the bubbles for the current question from left to right (A, B, C, D)
        question_bubbles = sorted(question_bubbles, key=lambda c: cv2.boundingRect(c)[0])
        
        # Find the filled-in bubble
        marked_bubble = None
        
        for j, bubble in enumerate(question_bubbles):
            # Calculate the density of dark pixels inside the bubble contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            
            # Count the number of black pixels within the mask
            pixels = cv2.countNonZero(mask)
            filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            
            # The percentage of filled pixels is our heuristic for a marked bubble
            filled_percentage = (filled_pixels / float(pixels)) * 100
            
            if filled_percentage > 50:  # If more than 50% is filled, it's a marked bubble
                marked_bubble = j  # 0 for A, 1 for B, etc.
                break
        
        # Map the marked bubble index to the corresponding letter
        if marked_bubble is not None:
            answer_letter = chr(ord('A') + marked_bubble)
            detected_answers[i + 1] = answer_letter
        else:
            detected_answers[i + 1] = 'Unanswered'
    
    # Score the test
    score = 0
    for question, answer in detected_answers.items():
        if question in ANSWER_KEY and answer == ANSWER_KEY[question]:
            score += 1
            
    return score, detected_answers

# --- Generative AI Logic ---

def generate_ai_feedback(answers, score):
    """
    Generates personalized feedback using the Gemini API.
    
    Args:
        answers (dict): The user's detected answers.
        score (int): The calculated score.
        
    Returns:
        str: A string containing the personalized feedback.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Construct a detailed prompt for the AI
    correct_answers = {q: a for q, a in answers.items() if q in ANSWER_KEY and a == ANSWER_KEY[q]}
    incorrect_answers = {q: a for q, a in answers.items() if q in ANSWER_KEY and a != ANSWER_KEY[q]}
    
    prompt = f"""
    You are an expert educational AI assistant. You have just evaluated an OMR test.
    The student's score is {score} out of {len(ANSWER_KEY)}.

    Here are the student's answers:
    {answers}
    
    Based on this, provide detailed, personalized feedback.
    - Start with a positive, encouraging statement about the student's effort.
    - Identify their strengths based on the correct answers.
    - Identify areas for improvement based on the incorrect or unanswered questions.
    - Give specific, actionable advice on how to improve.
    - Recommend study strategies.
    - Use a friendly and supportive tone.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating feedback: {str(e)}"

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page for the web app."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles the file upload, processes the OMR sheet, and returns the result.
    """
    if 'omr_image' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['omr_image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
        
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        # Process the OMR sheet
        score, detected_answers = process_omr_sheet(file_path)
        
        # Generate AI feedback
        ai_feedback = generate_ai_feedback(detected_answers, score)
        
        # Return the results
        return jsonify({
            "score": score,
            "total_questions": len(ANSWER_KEY),
            "detected_answers": detected_answers,
            "ai_feedback": ai_feedback,
            "answer_key": ANSWER_KEY
        })
        
    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500
        
    finally:
        # Clean up the temporary file
        os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)

