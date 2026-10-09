import json, csv

with open('data/manifests/identity_split.json') as f:
    split = json.load(f)

with open('data/manifests/samples.csv') as f:
    samples = list(csv.DictReader(f))

print(f"Train identities: {split['train']}")
print(f"Val identities: {split['val']}")
print(f"Test identities: {split['test']}")

train_img = sum(1 for s in samples if s['split']=='train')
val_img = sum(1 for s in samples if s['split']=='val')
test_img = sum(1 for s in samples if s['split']=='test')

print(f"Train images: {train_img}")
print(f"Val images: {val_img}")
print(f"Test images: {test_img}")
