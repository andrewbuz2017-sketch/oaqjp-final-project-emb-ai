# Emotion Detector

Flask application using the Watson NLP emotion service to return anger, disgust, fear, joy, sadness, and the dominant emotion. Empty input receives a clear error message.

## Structure

- EmotionDetection/__init__.py: package export
- EmotionDetection/emotion_detection.py: API client and response formatting
- server.py: Flask routes and application entry point
- test_emotion_detection.py: five classification tests
- templates and static: web interface from the IBM starter repository

## Run in the IBM Skills Network lab

```sh
pip install flask requests pylint
python -m unittest test_emotion_detection.py
pylint server.py
python server.py
```

Open the application on port 5000. The Watson endpoint requires access from the Skills Network lab.

## Validation

All five emotion tests passed. Pylint rated server.py 10.00/10. Valid and empty inputs were checked through the web interface. Evidence files and screenshots accompany this project.
