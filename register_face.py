from pathlib import Path
from datetime import datetime
import re

import cv2
import numpy as np
from arcface import ArcFace
from facenet_pytorch import MTCNN
from scipy.spatial.distance import cosine


EMBEDDINGS_PATH = Path("facebank_embeddings2.npy")
NAMES_PATH = Path("student_names2.npy")
PHOTOS_DIR = Path("class_student_photos")
MODEL_PATH = "model.tflite"
DUPLICATE_THRESHOLD = 0.4

_mtcnn = None
_face_model = None


def _ensure_models():
    global _mtcnn, _face_model
    if _mtcnn is None:
        _mtcnn = MTCNN(image_size=112, margin=0, keep_all=False, post_process=False)
    if _face_model is None:
        _face_model = ArcFace.ArcFace(model_path=MODEL_PATH)
    return _mtcnn, _face_model


def _safe_filename(name):
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "student"


def _detect_and_align_face(image_bgr, mtcnn):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    if face_tensor is None:
        return None
    face_rgb = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)


def _get_embedding(face_img, face_model):
    embedding = face_model.calc_emb(face_img)
    return np.asarray(embedding).flatten()


def _load_facebank():
    if EMBEDDINGS_PATH.exists() and NAMES_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        names = np.load(NAMES_PATH)
        if len(embeddings) != len(names):
            raise ValueError(
                f"Mismatch in facebank lengths: {len(embeddings)} embeddings vs {len(names)} names."
            )
        return embeddings, names
    return np.empty((0, 512), dtype=np.float32), np.empty((0,), dtype=object)


def _save_facebank(embeddings, names):
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(NAMES_PATH, names)


def register_face_from_image(image_bgr, student_name):
    student_name = str(student_name).strip()
    if not student_name:
        return False, "Please enter a student name."

    mtcnn, face_model = _ensure_models()
    face_img = _detect_and_align_face(image_bgr, mtcnn)
    if face_img is None:
        return False, "No face detected. Please capture a clear face photo."

    embedding = _get_embedding(face_img, face_model)
    if embedding.size == 0 or np.linalg.norm(embedding) == 0:
        return False, "Invalid face embedding. Try capturing again."

    facebank, names = _load_facebank()

    if len(facebank) > 0:
        distances = np.array([cosine(embedding, fb) for fb in facebank], dtype=np.float32)
        match_index = int(np.argmin(distances))
        min_distance = float(distances[match_index])
        if min_distance < DUPLICATE_THRESHOLD:
            existing = str(names[match_index])
            return (
                False,
                f"Already registered as '{existing}' (distance {min_distance:.3f}).",
            )

    safe_name = _safe_filename(student_name)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    photo_path = PHOTOS_DIR / f"{safe_name}.jpg"
    if photo_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = PHOTOS_DIR / f"{safe_name}_{stamp}.jpg"
    cv2.imwrite(str(photo_path), image_bgr)

    if len(facebank) == 0:
        updated_embeddings = np.array([embedding], dtype=np.float32)
        updated_names = np.array([student_name])
    else:
        updated_embeddings = np.vstack([facebank, embedding]).astype(np.float32)
        updated_names = np.append(names, student_name)

    _save_facebank(updated_embeddings, updated_names)
    return True, f"Registered '{student_name}' successfully."
