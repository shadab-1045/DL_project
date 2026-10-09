import hashlib
import os

EXPECTED_HASHES = {
    "Model A": {
        "path": r"C:\Users\shada\.insightface\models\buffalo_l\w600k_r50.onnx",
        "hash": "4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43"
    },
    "C-Control": {
        "path": "experiments/model_c_control/best_model.pt",
        "hash": "1CF70BA964B5E60C6632AC8532A7BA8BCEC9F4509E425CE9CE6DF71E00C86E51"
    },
    "C-Adv": {
        "path": "experiments/model_c_adv/best_model.pt",
        "hash": "A562487A5B3CC085E3CC1EED6FBEA93748376FB75BBD2FBDF3D0E871F966AFBD"
    }
}

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().upper()

for name, info in EXPECTED_HASHES.items():
    h = compute_sha256(info["path"])
    print(f"{name}: {h} (Expected: {info['hash']}) (Match: {h == info['hash']})")
