import os
import urllib.request

def download_image(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Downloading {url} to {save_path}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
    else:
        print(f"Already exists: {save_path}")

if __name__ == "__main__":
    base_dir = "sample_data"
    
    # Person A: Einstein
    download_image("https://upload.wikimedia.org/wikipedia/commons/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg", f"{base_dir}/P001_ref1.jpg")
    download_image("https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg", f"{base_dir}/P001_ref2.jpg")
    download_image("https://upload.wikimedia.org/wikipedia/commons/5/50/Albert_Einstein_%28Nobel%29.png", f"{base_dir}/P001_probe.jpg")
    
    # Person B: Marie Curie
    download_image("https://upload.wikimedia.org/wikipedia/commons/c/c8/Marie_Curie_c._1920s.jpg", f"{base_dir}/P002_ref1.jpg")
    download_image("https://upload.wikimedia.org/wikipedia/commons/7/7e/Marie_Curie_c1920.png", f"{base_dir}/P002_probe.jpg")

    print("Sample dataset downloaded.")
