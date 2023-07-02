import os
import cv2
import joblib

decision_tree_model = joblib.load('rock_paper_scissor_decision_tree.joblib')
random_forest_model = joblib.load('rock_paper_scissor_random_forest.joblib')
knn_model = joblib.load('knn.joblib')
gaussian_nb_model = joblib.load('gnb.joblib')

filename = 'scissor2.png'
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
knn_y_pred = knn_model.predict([hog_features])[0]
gaussian_nb_y_pred = gaussian_nb_model.predict([hog_features])[0]


def predict(modelName, value):
    if value == 0:
        print(modelName, filename, 'Rock')
    elif value == 1:
        print(modelName, filename, 'Paper')
    elif value == 2:
        print(modelName, filename, 'Scissor')
    else:
        print('Unknown value', dt_y_pred)

predict('DT', dt_y_pred)
predict('RF', rf_y_pred)
predict('KNN', knn_y_pred)
predict('GNB', gaussian_nb_y_pred)