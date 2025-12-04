import requests
import os

API_URL = "http://127.0.0.1:5000/classify"

IMAGE_PATH = "tomat-busuk.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at '{IMAGE_PATH}'")
    print("Please make sure the image file is in the same directory as the script.")
    exit()

try:
    with open(IMAGE_PATH, 'rb') as image_file:
        files = {'image': image_file}
        print(f"Sending '{IMAGE_PATH}' to {API_URL}...")
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        print("\nPrediction successful!")
        print("Response JSON:", response.json())
    else:
        print(f"\nError: Server returned status code {response.status_code}")
        try:
            print("Response JSON:", response.json())
        except requests.exceptions.JSONDecodeError:
            print("Response content:", response.text)

except requests.exceptions.ConnectionError as e:
    print(f"\nError: Could not connect to the server at {API_URL}.")
    print("Please make sure the Flask app ('app.py') is running.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
