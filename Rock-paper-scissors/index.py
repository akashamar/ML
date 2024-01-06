import os
import cv2
import joblib
import numpy as np
import tensorflow as tf

with tf.device('/CPU:0'):
    decision_tree_model = joblib.load('rock_paper_scissor_decision_tree.joblib')
    random_forest_model = joblib.load('rock_paper_scissor_random_forest.joblib')
    knn_model = joblib.load('knn.joblib')
    gaussian_nb_model = joblib.load('gnb.joblib')
    svc_model = joblib.load('svc.joblib')
    kmeans_model = joblib.load('kmeans.joblib')
    neural_network = joblib.load('neural_network.joblib')

image_width = 128  # Desired width
image_height = 128  # Desired height

# Function to extract HOG features from an image
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    hog_features = hog_features.reshape(-1)  # Flatten the feature vector
    return hog_features

# Function to predict the class label using a given model
def predict(modelName, value):
    if value == 0:
        print(modelName, 'Rock')
    elif value == 1:
        print(modelName, 'Paper')
    elif value == 2:
        print(modelName, 'Scissor')
    else:
        print('Unknown value')

def capture_image():
    print('capturing')
    # Open the webcam for capturing video
    cap = cv2.VideoCapture(0)
    # Capture a single image from the webcam
    ret, frame = cap.read()
    # Resize the frame
    frame = cv2.resize(frame, (image_width, image_height))
    # Save the captured image
    filename = 'captured_image.png'
    cv2.imwrite(filename, frame)
    print(f'Saved captured image as {filename}')
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return gray

def makePrediction(hog_features):
    # predict
    dt_y_pred = decision_tree_model.predict([hog_features])[0]
    rf_y_pred = random_forest_model.predict([hog_features])[0]
    knn_y_pred = knn_model.predict([hog_features])[0]
    gaussian_nb_y_pred = gaussian_nb_model.predict([hog_features])[0]
    svc_y_pred = svc_model.predict([hog_features])[0]
    km_y_pred = kmeans_model.predict(hog_features.reshape(1, -1))[0]
    neural_net_y_pred = neural_network.predict(hog_features.reshape(1, -1))[0]
    nn_y_pred = np.argmax(neural_net_y_pred)
    print('neural_net_y_pred_values', neural_net_y_pred)
    predict('neural_net_y_pred', nn_y_pred)
    # Print the predictions for each model
    predict('Decision Tree', dt_y_pred)
    predict('Random Forest', rf_y_pred)
    predict('K Nearest Neighbor', knn_y_pred)
    predict('Gaussian Naive Bayes', gaussian_nb_y_pred)
    predict('Support Vector Machine', svc_y_pred)
    predict('K-Means Clustering', km_y_pred)
    predictions = [dt_y_pred, rf_y_pred, knn_y_pred, gaussian_nb_y_pred, svc_y_pred, km_y_pred]
    return predictions

def finalPrediction(predictions):
    # Count the votes for each class
    votes = [predictions.count(0), predictions.count(1), predictions.count(2)]
    # Get the index of the class with the most votes
    final_prediction = np.argmax(votes)
    # Print the final prediction
    predict('Final Output', final_prediction)

def main():
    image = cv2.imread(os.path.join('./', "scissor2.png"))
    gray = cv2.resize(image, (image_width, image_height))  # Resize the image
    # gray = capture_image()
    # Extract HOG features from the grayscale frame
    hog_features = extract_hog_features(gray)
    # Make predictions using the loaded models
    predictions = makePrediction(hog_features)
    finalPrediction(predictions)

main()