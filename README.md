# automated-omr-evaluator
automated-omr-evaluator
The Problem
Manual grading of OMR (Optical Mark Recognition) sheets is a time-consuming and error-prone process for educators. It also provides minimal feedback to students, who only see a final score without understanding where they went wrong or why.

Our Solution
Our "Automated OMR Evaluation & Scoring System" is a web application that uses computer vision to instantly grade OMR sheets from a simple photo. It then leverages generative AI to provide personalized, detailed feedback to students, transforming a simple score into a valuable learning experience.

Key Features (Minimum Viable Product)
Image Upload: Users can upload a photo of a student's OMR sheet.

Automatic Alignment: The system automatically detects the corners of the sheet and corrects for any perspective distortion.

Bubble Detection: It identifies all the bubbles on the sheet, even if they are slightly off-center.

Answer Evaluation: The system accurately reads the marked bubbles and compares them to a pre-defined answer key.

Score Calculation: The final score is calculated and displayed in real-time.

Generative AI Feedback: A personalized paragraph of feedback is generated for the student, highlighting their strengths and areas for improvement.

Technologies Used
Frontend: HTML, CSS, JavaScript (for the web interface)

Backend: Python (for the core logic)

Computer Vision: OpenCV (to process the image)

Generative AI: The Gemini API (to create personalized feedback)
