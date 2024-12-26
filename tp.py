import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)
print(f"Encryption Key: {key.decode()}")

# Folder containing fingerprint images
folder_path = "Dataset/test"

# Process each .bmp image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")

        # Load the fingerprint image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Show the processed image using matplotlib
        plt.imshow(thresholded, cmap='gray')
        plt.title('Processed Fingerprint')
        plt.show()

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(thresholded, None)

        # Draw keypoints on the image
        output_image = cv2.drawKeypoints(thresholded, keypoints, None)
        plt.imshow(output_image, cmap='gray')
        plt.title('Keypoints')
        plt.show()

        # Convert descriptors to bytes and encrypt them
        if descriptors is not None:
            descriptors_bytes = descriptors.tobytes()
            encrypted_data = cipher_suite.encrypt(descriptors_bytes)

            # Save the encrypted descriptors to a file
            encrypted_file_path = os.path.join(folder_path, f"{filename}_encrypted.dat")
            with open(encrypted_file_path, 'wb') as file:
                file.write(encrypted_data)
            print(f"Encrypted descriptors saved to {encrypted_file_path}")

        else:
            print(f"No descriptors found for {filename}, skipping encryption.")

# Matching and performance evaluation
def evaluate_matching(matches):
    threshold_ratio = 0.7
    total_matches = len(matches)
    true_positives = sum(1 for m in matches if m.distance < threshold_ratio)
    false_positives = total_matches - true_positives
    y_true = [1] * true_positives + [0] * false_positives
    y_pred = [1] * true_positives + [0] * false_positives
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

print("Processing completed.")
