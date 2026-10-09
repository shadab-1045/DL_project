# Data and Dataset Specification

## 1. Data roles

| Data source | Role |
|---|---|
| VGGFace2 | identity-organized public face data |
| FaceForensics++ | real/manipulated video, including FaceSwap |
| Controlled generated swaps | project-specific source/target impersonation samples |
| Deep-Live-Cam | live manipulation evaluation |
| Consenting participants | final live enrollment/demo only |

Verify access/licensing before use or redistribution.

## 2. VGGFace2
Use for identity-labeled data and source/target selection. The project does not require the full dataset.

Initial study scale can be configurable, for example 150–300 identities and 20–50 usable images per identity, adjusted after storage/data-quality checks.

Reference:
https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

## 3. FaceForensics++
Use FaceSwap manipulation data as additional training/evaluation evidence and for external validation. Follow its official metadata and target/source naming rather than inferring relationships.

Reference:
https://github.com/ondyari/FaceForensics

## 4. Controlled swaps
Select identities from the identity dataset and generate controlled swaps:
- P001-on-P002
- P001-on-P003
- P002-on-P001
- etc.

Use SimSwap or an equivalent controlled generator.

Reference:
https://github.com/neuralchen/SimSwap

## 5. Labels
Internal semantic labels:
- `genuine`
- `different_person`
- `impersonation`

FGSM/PGD variants of impersonation retain `impersonation` as their semantic label; metadata stores the attack type.

## 6. Key metadata
`samples.csv`:
```text
sample_id
path
visible_identity
physical_identity
attack_type
source_dataset
generator
video_id
frame_id
split
```

Pair manifest:
```text
pair_id
reference_identity
reference_embedding_path
probe_sample_id
probe_path
visible_identity
physical_identity
label
attack_type
source_dataset
generator
split
```

## 7. Example
```csv
T0001,P001,gallery/P001/embedding.npy,S0001,faces/P001/01.jpg,P001,P001,genuine,none,vggface2,none,train
T0002,P001,gallery/P001/embedding.npy,S0101,faces/P002/01.jpg,P002,P002,different_person,none,vggface2,none,train
T0003,P001,gallery/P001/embedding.npy,S0201,swaps/P001_on_P002/01.jpg,P001,P002,impersonation,face_swap,generated,simswap,train
```

## 8. Data relationship
For reference Alice/P001:
- genuine Alice → `genuine`
- genuine Bob → `different_person`
- Alice-on-Bob → `impersonation`

The `physical_identity` field captures who actually appears in the original source/target context when known.

## 9. Dataset generation
```text
public identity data
  ↓
identity subset
  ↓
face preprocessing
  ↓
controlled source/target selection
  ↓
face-swap generation
  ↓
metadata
  ↓
hard transformations
  ↓
optional FGSM/PGD
  ↓
manifests
```

## 10. Split policy
Perform identity/video-level splits before pair generation. Never split frames from the same source video across train/validation/test. Keep live demo users out of training.

## 11. Data QA
Check:
- unreadable files
- duplicate leakage
- valid face detection
- correct identity metadata
- correct swap direction
- class balance
- split leakage
- random visual samples
