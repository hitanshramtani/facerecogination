import os
import cv2
import numpy as np
from glob import glob
from datetime import datetime
from pathlib import Path
from arcface import ArcFace
from facenet_pytorch import MTCNN
from scipy.spatial.distance import cosine
from openpyxl import Workbook, load_workbook

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
model = ArcFace.ArcFace(model_path="model.tflite")
mtcnn = MTCNN(image_size=112, margin=0, keep_all=False, post_process=False)
ATTENDANCE_FILE = Path("Attendance_sheet_demo1.xlsx")
PRESENT_VALUE = "P"
ABSENT_VALUE = "A"
MARKED_TODAY = set()

def load_facebank_and_names():
    candidate_pairs = [
        ("facebank_embeddings2.npy", "student_names2.npy"),
        ("facebank_embeddings.npy", "student_names.npy"),
    ]
    for emb_path, names_path in candidate_pairs:
        if Path(emb_path).exists() and Path(names_path).exists():
            facebank_arr = np.load(emb_path)
            names_arr = np.load(names_path)
            if len(facebank_arr) != len(names_arr):
                raise ValueError(
                    f"Count mismatch in {emb_path} and {names_path}: "
                    f"{len(facebank_arr)} vs {len(names_arr)}"
                )
            print(
                f"Loaded face bank: {emb_path} shape={facebank_arr.shape}, "
                f"labels: {names_path} shape={names_arr.shape}"
            )
            return facebank_arr, names_arr
    raise FileNotFoundError(
        "Could not find face bank files. Expected either "
        "(facebank_embeddings2.npy, student_names2.npy) or "
        "(facebank_embeddings.npy, student_names.npy)."
    )

facebank, student_names = load_facebank_and_names()

def _get_april_headers(year):
    return [f"{day:02d}-Apr-{year}" for day in range(1, 31)]

def _find_student_row(sheet, student_name):
    normalized = student_name.strip().lower()
    for row_idx in range(2, sheet.max_row + 1):
        existing = sheet.cell(row=row_idx, column=1).value
        if existing and str(existing).strip().lower() == normalized:
            return row_idx
    return None

def _get_or_create_date_column(sheet, date_label):
    for col_idx in range(2, sheet.max_column + 1):
        header = sheet.cell(row=1, column=col_idx).value
        if str(header).strip() == date_label:
            return col_idx
    new_col = sheet.max_column + 1
    sheet.cell(row=1, column=new_col, value=date_label)
    for row_idx in range(2, sheet.max_row + 1):
        sheet.cell(row=row_idx, column=new_col, value=ABSENT_VALUE)
    return new_col

def initialize_attendance_sheet(all_students):
    if ATTENDANCE_FILE.exists():
        return

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Attendance"
    year = datetime.now().year
    headers = ["Student Name", *_get_april_headers(year)]
    sheet.append(headers)

    for student in sorted({str(name).strip() for name in all_students if str(name).strip()}):
        sheet.append([student] + [ABSENT_VALUE] * (len(headers) - 1))

    workbook.save(ATTENDANCE_FILE)
    workbook.close()
    print(f"Created attendance sheet: {ATTENDANCE_FILE}")

def detect_and_align_face(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    
    if face_tensor is None:
        return None

    # MTCNN gives CHW RGB crop; convert to HWC BGR for ArcFace wrapper.
    face_rgb = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    return face_bgr

def get_embedding(face_img):
    # ArcFace wrapper already handles RGB conversion, resize, and normalization.
    embedding = model.calc_emb(face_img)
    return np.asarray(embedding).flatten()

def mark_attendance(student_name):
    student_name = str(student_name).strip()
    if not student_name:
        return False, "Invalid student name"

    now = datetime.now()
    date_str = now.strftime("%d-%b-%Y")
    cache_key = (student_name.lower(), date_str)
    if cache_key in MARKED_TODAY:
        return False, f"{student_name}: attendance already marked for {date_str}"

    workbook = load_workbook(ATTENDANCE_FILE)
    sheet = workbook.active

    student_row = _find_student_row(sheet, student_name)
    if student_row is None:
        student_row = sheet.max_row + 1
        sheet.cell(row=student_row, column=1, value=student_name)
        for col_idx in range(2, sheet.max_column + 1):
            sheet.cell(row=student_row, column=col_idx, value=ABSENT_VALUE)

    date_col = _get_or_create_date_column(sheet, date_str)
    status_cell = sheet.cell(row=student_row, column=date_col)
    already_marked = str(status_cell.value).strip().upper() == PRESENT_VALUE
    if already_marked:
        MARKED_TODAY.add(cache_key)
        workbook.close()
        return False, f"{student_name}: attendance already marked for {date_str}"

    status_cell.value = PRESENT_VALUE
    workbook.save(ATTENDANCE_FILE)
    workbook.close()
    MARKED_TODAY.add(cache_key)
    return True, f"{student_name}: attendance marked present for {date_str}"
    
def recognize_face(embedding, facebank, threshold=0.4):
    # Calculate cosine similarity between embedding and each face in the facebank
    distances = [cosine(embedding, fb) for fb in facebank]
    min_distance = np.min(distances)
    match_index = np.argmin(distances)
    if min_distance < threshold:
        return student_names[match_index], min_distance
    return None, None

initialize_attendance_sheet(student_names)

# Initialize webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_img = detect_and_align_face(frame)
    if face_img is None:
        cv2.putText(frame, "No Face Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    embedding = get_embedding(face_img)
    
    # Recognize face by comparing with facebank
    student, dist = recognize_face(embedding, facebank)
    if student:
        was_marked, status_msg = mark_attendance(student)
        cv2.putText(frame, f"{student} ({dist:.2f})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        status_color = (0, 255, 0) if was_marked else (0, 255, 255)
        status_text = "Attendance Marked" if was_marked else "Attendance Already Marked"
        cv2.putText(frame, status_text, (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        print(status_msg)
    
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
