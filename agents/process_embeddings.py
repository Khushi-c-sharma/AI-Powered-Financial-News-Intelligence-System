# process_embeddings.py
import requests
import time

while True:
    response = requests.post("http://localhost:8002/embed/process?limit=50")
    print(response.json())
    time.sleep(60)  # Process every minute