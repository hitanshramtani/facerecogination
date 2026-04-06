"""
Generic YOLO models are trained to detect classes like "person", "car", etc.
YOLO just gives you bounding boxes — it won't align faces like MTCNN or RetinaFace would.
ArcFace works best with aligned faces.
 You might want to apply facial landmark detection after detection to align
YOLO might miss tiny or angled faces unless it's trained to handle them.
"""

import os
import cv2
import numpy as np
from glob import glob
from datetime import datetime
from arcface import ArcFace
from facenet_pytorch import MTCNN
import torch


mtcnn = MTCNN(image_size=112, margin=0, keep_all=False)
face_rec = ArcFace.ArcFace(model_path="model.tflite")


def detect_and_align_face(image):
    """Detect and align face using MTCNN. Returns aligned face or None."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    
    if face_tensor is None:
        return None

    face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
    return face_np

def preprocess_face(image):
    aligned_face = detect_and_align_face(image)
    if aligned_face is None:
        return None
    try:
        face_img = cv2.resize(aligned_face, (112, 112))
        return face_img
    except Exception as e:
        print(f"[ERROR] Failed to resize face image: {e}")
        return aligned_face


def get_embedding(face_img):
    # Normalize image to float32 in range [0, 1]
    face_img = face_img.astype('float32') / 255.0
    #model input shape (1,112,112,3)
    face_img = np.expand_dims(face_img, axis=0)
    # Here, model.predict(face_img)[0] would be your 512-d vector.
    embedding = face_rec.calc_emb(face_img[0])
    return embedding


facebank = []
labels = []

root_dataset_dir = r"C:\Users\hitan\Desktop\coding\dl\face recogination through cnn\Dataset\final_dataset"
#main loop
for batch_name in os.listdir(root_dataset_dir):

    #loop through each batch folder
    dataset_dir = os.path.join(root_dataset_dir, batch_name)
    i = 1
    for student_folder in os.listdir(dataset_dir):
        
        # Loop through each student folder
        print(f"Student{i}")
        i = i+1
        student_path = os.path.join(dataset_dir, student_folder)
        if not os.path.isdir(student_path):
            continue

        embeddings = []

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            face_img = preprocess_face(image)
            if face_img is None:
                print(f"No face detected in: {img_path}")
                continue

            emb = get_embedding(face_img)
            embeddings.append(emb)

        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            facebank.append(mean_embedding)
            labels.append(student_folder)
            print(f"Added '{student_folder}' with {len(embeddings)} samples.")
        else:
            print(f"No valid images for '{student_folder}'")


# Convert to numpy arrays
facebank = np.array(facebank)
labels = np.array(labels)

# Save the facebank and labels for later use
np.save("facebank_embeddings.npy", facebank)
np.save("student_names.npy", labels)

print("Face bank created successfully!")
