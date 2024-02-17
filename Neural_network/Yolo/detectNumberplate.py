from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv('ROBO_API_KEY')
)

result = CLIENT.infer("https://media.geeksforgeeks.org/wp-content/uploads/20200326001003/blurred.jpg", model_id="carplate-xuk6s/1")
print(result)