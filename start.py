import cv2
from arcface import ArcFace
import numpy as np
import os

# Function to check if an image is loaded correctly
def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image - {image_path}")
    return img

# Try to load ArcFace with error handling
try:
    face_rec = ArcFace.ArcFace(model_path="model.tflite")

except Exception as e:
    print("Error loading ArcFace:", e)
    exit()

image_path1 = r"C:\Users\hitan\Desktop\coding\dl\face recogination through cnn\HITANSH RAMTANI.jpg"
image_path2 = r"C:\Users\hitan\Desktop\coding\dl\face recogination through cnn\HITANSH RAMTANI.jpg"

img1 = load_image(image_path1)
if img1 is None:
    exit()
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

#embedding for first image
embedding1 = face_rec.calc_emb(img1_rgb)
print("Face embedding 1:", embedding1)

img2 = load_image(image_path2)
if img2 is None:
    exit()
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#embedding for second image
embedding2 = face_rec.calc_emb(img2_rgb)
print("Face embedding 2:", embedding2)

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        # print("Error: Zero-vector embedding detected.")
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

# Compute similarity
distance = cosine_similarity(embedding1, embedding2)
print("Cosine similarity (higher means more similar):", distance)