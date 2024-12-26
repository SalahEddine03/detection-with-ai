import cv2
import numpy as np
from cryptography.fernet import Fernet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)
print(f"Encryption Key: {key.decode()}")

# Load the fingerprint image
image = cv2.imread('finger.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocess the image
blurred = cv2.GaussianBlur(image, (5, 5), 0)
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show the processed image
cv2.imshow('Processed Fingerprint', thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(thresholded, None)

# Draw keypoints on the image
output_image = cv2.drawKeypoints(thresholded, keypoints, None)
cv2.imshow('Keypoints', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert descriptors to bytes and encrypt them
descriptors_bytes = descriptors.tobytes()
encrypted_data = cipher_suite.encrypt(descriptors_bytes)

# Store or transmit the encrypted data securely
with open('encrypted_fingerprint.dat', 'wb') as file:
    file.write(encrypted_data)

    # Load encrypted fingerprint data from the file
with open('encrypted_fingerprint.dat', 'rb') as file:
    encrypted_data = file.read()

# Decrypt the data
decrypted_data = cipher_suite.decrypt(encrypted_data)

# Convert bytes back to numpy array for matching (reshape if necessary)
decrypted_descriptors = np.frombuffer(decrypted_data, dtype=np.float32).reshape(-1,
                                                                                128)  # Adjust shape based on your descriptor size

# Match descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors, decrypted_descriptors)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Display matching results (if needed)
matching_result = cv2.drawMatches(image, keypoints, image, keypoints, matches[:10], None)
cv2.imshow('Matches', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Evaluate matching performance using metrics
def evaluate_matching(matches):
    # Assuming a threshold for determining a match (e.g., 0.7 ratio of matches)
    threshold_ratio = 0.7
    total_matches = len(matches)

    # Calculate true positives (TP) and false positives (FP) based on some criteria.
    # Here we assume if matches are above a certain distance they are considered valid.
    true_positives = sum(1 for m in matches if m.distance < threshold_ratio)
    false_positives = total_matches - true_positives

    # Dummy values for actual labels (1 for match, 0 for no match)
    y_true = [1] * true_positives + [0] * false_positives
    y_pred = [1] * true_positives + [0] * false_positives

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


accuracy, precision, recall, f1 = evaluate_matching(matches)

print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')