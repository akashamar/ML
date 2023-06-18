import os
import cv2
import joblib

model = joblib.load('rock_paper_scissor_decision_tree.joblib')

filename = 'scissor3.png'
image_width = 128  # Desired width
image_height = 128  # Desired height

# Function to extract HOG features from an image
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    hog_features = hog_features.reshape(-1)  # Flatten the feature vector
    return hog_features

image = cv2.imread(os.path.join('./', filename))
image = cv2.resize(image, (image_width, image_height))  # Resize the image
hog_features = extract_hog_features(image)

y_pred = model.predict([hog_features])[0]

print(y_pred)
if y_pred == 0:
    print(filename, 'Rock')
elif y_pred == 1:
    print(filename, 'Paper')
elif y_pred == 2:
    print(filename, 'Scissor')
else:
    print('Unknown value', y_pred)
