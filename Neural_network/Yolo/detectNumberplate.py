from roboflow import Roboflow
import supervision as sv
from dotenv import load_dotenv
import os
import cv2

load_dotenv()

rf = Roboflow(api_key=os.getenv('ROBO_API_KEY'))
project = rf.workspace().project("carplate-xuk6s")
model = project.version(1).model

result = model.predict("blurred.jpg", confidence=40, overlap=30).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_inference(result)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread("blurred.jpg")

annotated_image = label_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))