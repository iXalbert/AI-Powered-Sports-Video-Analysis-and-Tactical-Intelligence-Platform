import os
import urllib.request

os.makedirs("models", exist_ok=True)
target_path = os.path.join("models", "football_yolov8.pt")
url = "https://huggingface.co/uisikdag/yolo-v8-football-players-detection/resolve/main/best.pt"

if not os.path.exists(target_path):
    print(f"Descărcare model fine-tuned fotbal în {target_path}...")
    urllib.request.urlretrieve(url, target_path)
    print("Descărcare finalizată cu succes!")
else:
    print(f"Modelul există deja la {target_path}.")