import requests
import os

url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
dest = "D:\\COLLEGE\\sem-5\\Deep_Learning\\CP\\adverserial_project\\models\\inswapper_128.onnx"

print(f"Downloading {url} to {dest}...")
response = requests.get(url, stream=True)
response.raise_for_status()

with open(dest, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
print("Download complete.")
