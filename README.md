
# Three Stage Authentication System for Cabin Security

This project implements a secure biometric authentication system for smart cabin environments, incorporating fingerprint mapping, facial recognition, and speaker verification. It uses Raspberry Pi as the central controller, with each biometric method being integrated for enhanced security.

## Features:
- **Fingerprint Verification**: Captures and verifies fingerprint data using the R307 optical sensor, with template matching via SIFT and Flann-based matcher.
- **Face Verification**: Utilizes OpenCV and DeepFace for detecting and verifying facial features with ResNet and FaceNet models.
- **Speaker Verification**: Implements voice authentication through MFCC feature extraction, cosine similarity comparison, and verification.
  
## Components Used:
- Raspberry Pi 4 Model B
- R307 Fingerprint Sensor
- Raspberry Pi Camera Module V2
- Microphone and Speaker for voice input/output

## Tools and Libraries:
- Python with Tkinter for GUI
- OpenCV, DeepFace, and Librosa for face and voice recognition
- SIFT and Flann-based matcher for fingerprint verification

## Results:
The system successfully integrates three-stage biometric authentication to provide robust security for a smart cabin, offering a comprehensive voting scheme across all three biometric methods.

## Future Enhancements:
- Real-time processing improvements
- Multi-language speaker verification
- Expanded biometric methods
