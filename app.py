import os
import cv2
import numpy as np
from glob import glob
from datetime import datetime
from arcface import ArcFace
from facenet_pytorch import MTCNN
from scipy.spatial.distance import cosine

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
model = ArcFace.ArcFace(model_path="model.tflite")
mtcnn = MTCNN(image_size=112, margin=0, keep_all=False)

facebank = np.load('facebank_embeddings.npy')
student_names = np.load('student_names.npy')

def detect_and_align_face(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    
    if face_tensor is None:
        return None

    # Keep float values from MTCNN. Casting to int collapses information and
    # can make different faces look artificially similar to the recognizer.
    face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
    return face_np

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (112, 112))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    embedding = model.calc_emb(face_img[0])
    return np.asarray(embedding).flatten()

def mark_attendance(student_name):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%d-%B-%Y')
    with open("Attendance_sheet_demo1.txt", "a") as f:
        f.write(f"{student_name},{time_str},{date_str}\n")
    print(f"Marked {student_name} as present at {time_str} on {date_str}.")
    
def recognize_face(embedding, facebank, threshold=0.4):
    # Calculate cosine similarity between embedding and each face in the facebank
    distances = [cosine(embedding, fb) for fb in facebank]
    min_distance = np.min(distances)
    match_index = np.argmin(distances)
    if min_distance < threshold:
        return student_names[match_index], min_distance
    return None, None

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
        mark_attendance(student)
        cv2.putText(frame, f"{student} ({dist:.2f})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
