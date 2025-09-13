# Lie Detection System

This is a real-time lie detection system that analyzes facial expressions and voice patterns to detect potential deception. The system uses modern computer vision and audio processing techniques to provide accurate results.

## Features

- Real-time facial expression analysis
- Voice pitch analysis
- Webcam integration
- User-friendly interface
- Detailed analysis metrics

## Requirements

- Python 3.8 or higher
- Webcam
- Microphone
- Internet connection (for first-time setup)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd lie-detection-system
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run lie_detector.py
```

2. The application will open in your default web browser.

3. Use the interface:
   - Click "Start Recording" to begin the analysis
   - Speak naturally while looking at the webcam
   - Click "Stop Recording" when finished
   - Click "Analyze" to get the results

## How it Works

The system analyzes several factors to determine potential deception:

### Facial Analysis
- Eye movement and blinking patterns
- Mouth movements and expressions
- Head tilt and orientation
- Looking direction

### Voice Analysis
- Pitch variations
- Voice modulation
- Speech patterns

## Notes

- Ensure good lighting for accurate facial analysis
- Speak clearly and at a normal volume
- Maintain eye contact with the camera for best results
- The system works best in a quiet environment

## Disclaimer

This system is for educational and entertainment purposes only. The results should not be used as definitive proof of deception in any legal or professional context. 