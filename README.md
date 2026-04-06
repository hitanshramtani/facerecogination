# Face Recognition Attendance (ArcFace + MTCNN)

This project marks attendance from a live webcam feed using:
- `MTCNN` for face detection/alignment
- `ArcFace` (`model.tflite`) for face embeddings
- Cosine distance matching against a saved face bank

## Project Files

- `face_bank_creation.py`: Builds `facebank_embeddings.npy` and `student_names.npy` from dataset images.
- `app.py`: Main live attendance app (webcam + recognition + attendance logging).
- `test.py`: Debug version with detailed print logs (distances, embeddings, stats).
- `start.py`: Small sanity test for embedding similarity between two images.
- `progress.py`: Development notes and debugging history.
- `Attendance_sheet_demo1.txt`: Attendance output file.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout (Expected by `face_bank_creation.py`)

`face_bank_creation.py` expects this structure:

```text
Dataset/final_dataset/
  <batch_name>/
    <student_folder_name>/
      image1.jpg
      image2.jpg
      ...
```

Each `student_folder_name` is used as the label saved in `student_names.npy`.

## How To Run

1. Create the face bank:

```bash
python face_bank_creation.py
```

This generates:
- `facebank_embeddings.npy`
- `student_names.npy`

2. Start live attendance:

```bash
python app.py
```

Press `q` to quit the webcam window.

3. Optional debugging:

```bash
python test.py
```

4. Optional quick embedding check:

```bash
python start.py
```

## Notes

- `app.py` currently uses `cv2.VideoCapture(1)`. If camera does not open, change it to `0`.
- Matching in `app.py` uses cosine distance with threshold `0.4` (lower distance means better match).
- Attendance is appended to `Attendance_sheet_demo1.txt` as:
  `student_name,time,date`

## Required Local Files

Keep these in project root before running:
- `model.tflite`
- `facebank_embeddings.npy` and `student_names.npy` (generated after face bank creation)

