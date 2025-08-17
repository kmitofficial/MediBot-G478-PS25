# upload.py
import requests

url = "http://13.232.191.178:5000/upload"
file_path = "myfile.jpg"

files = {'image': open(file_path, 'rb')}
params = {
    "upload_id": "c9a2e1d6-7b5c-4d23-b81f-59b69ac2d0a7"
}

response = requests.post(url, files=files, data=params)

print("Status Code:", response.status_code)
print("Response:", response.text)
