import os
import cv2
import joblib

decision_tree_model = joblib.load('rock_paper_scissor_decision_tree.joblib')
random_forest_model = joblib.load('rock_paper_scissor_random_forest.joblib')

filename = 'rock3.png'
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

dt_y_pred = decision_tree_model.predict([hog_features])[0]
rf_y_pred = random_forest_model.predict([hog_features])[0]

if dt_y_pred == 0:
    print('DT', filename, 'Rock')
elif dt_y_pred == 1:
    print('DT', filename, 'Paper')
elif dt_y_pred == 2:
    print('DT', filename, 'Scissor')
else:
    print('Unknown value', dt_y_pred)

if rf_y_pred == 0:
    print('RF', filename, 'Rock')
elif rf_y_pred == 1:
    print('RF', filename, 'Paper')
elif rf_y_pred == 2:
    print('RF', filename, 'Scissor')
else:
    print('Unknown value', rf_y_pred)