# Face and Fingerprint Detection Project

## Overview
This project focuses on detecting faces and fingerprints using Python. It leverages image processing techniques to analyze images and perform detection tasks.

## Project Files

### Detection Scripts
- **`face_with_camera_detection.py`**: Detects faces using a live camera feed.
- **`face_with_picture_detection.py`**: Detects faces in an image.
- **`fingerprint_detection.py`**: Detects fingerprints in images.

### Data Files
- **`encrypted_fingerprint.dat`**: A sample encrypted fingerprint file used for testing fingerprint detection.
- **`faces.jpg`**: A sample image containing multiple faces for testing face detection.
- **`face.jpg`**: A single face image for testing.
- **`finger.jpg`**: A sample fingerprint image for detection.

## Requirements
To run this project, you need the following:
- Python 3.x
- OpenCV
- NumPy

Install the required Python packages using:
```bash
pip install -r requirements.txt
```

## How to Use

### Face Detection
1. **From Camera**:
   Run the script to detect faces in real-time using a camera:
   ```bash
   python face_with_camera_detection.py
   ```

2. **From an Image**:
   Run the script to detect faces in an image:
   ```bash
   python face_with_picture_detection.py
   ```

### Fingerprint Detection
Detect fingerprints from the provided image:
```bash
python fingerprint_detection.py
```

### Testing Encrypted Fingerprint File
This project also includes functionality to analyze the `encrypted_fingerprint.dat` file. Future development may include decryption and advanced analysis.

## Example Output

### Face Detection
**Input**: `faces.jpg`
**Output**: Image with detected faces highlighted by rectangles.

### Fingerprint Detection
**Input**: `finger.jpg`
**Output**: Processed image with fingerprint details highlighted.

## Future Enhancements
- Improve accuracy of fingerprint detection.
- Add support for real-time fingerprint scanning.
- Incorporate additional machine learning models for face and fingerprint recognition.

## License
This project is licensed under the MIT License.

## Contributions
Contributions are welcome! Feel free to fork this repository, create issues, and submit pull requests.

## Contact
For any questions or feedback, please reach out to the project maintainer.
