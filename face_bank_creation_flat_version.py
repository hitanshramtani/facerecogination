import argparse
from pathlib import Path

import cv2
import numpy as np
from arcface import ArcFace
from facenet_pytorch import MTCNN


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create face bank from a flat folder where each image is one student."
    )
    parser.add_argument(
        "--input-dir",
        default="class_student_photos",
        help="Flat folder containing student photos.",
    )
    parser.add_argument(
        "--model-path",
        default="model.tflite",
        help="Path to ArcFace TFLite model.",
    )
    parser.add_argument(
        "--embeddings-out",
        default="facebank_embeddings2.npy",
        help="Output path for face embeddings (.npy).",
    )
    parser.add_argument(
        "--labels-out",
        default="student_names2.npy",
        help="Output path for student labels (.npy).",
    )
    return parser.parse_args()


def detect_and_align_face(image_bgr, mtcnn):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    if face_tensor is None:
        return None
    face_rgb = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)


def get_embedding(face_img, face_model):
    embedding = face_model.calc_emb(face_img)
    return np.asarray(embedding).flatten()


def iter_images(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    mtcnn = MTCNN(image_size=112, margin=0, keep_all=False, post_process=False)
    face_model = ArcFace.ArcFace(model_path=str(model_path))

    facebank = []
    labels = []

    image_paths = list(iter_images(input_dir))
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")

    print(f"Found {len(image_paths)} image(s) in '{input_dir}'.")

    for idx, img_path in enumerate(image_paths, start=1):
        label = img_path.stem.strip()
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{idx}] Skipped (failed to load): {img_path.name}")
            continue

        face_img = detect_and_align_face(image, mtcnn)
        if face_img is None:
            print(f"[{idx}] Skipped (no face detected): {img_path.name}")
            continue

        emb = get_embedding(face_img, face_model)
        if emb.size == 0 or np.linalg.norm(emb) == 0:
            print(f"[{idx}] Skipped (invalid embedding): {img_path.name}")
            continue

        facebank.append(emb)
        labels.append(label)
        print(f"[{idx}] Added '{label}' from {img_path.name}")

    if not facebank:
        raise RuntimeError("No valid face embeddings were created.")

    facebank_arr = np.array(facebank)
    labels_arr = np.array(labels)

    np.save(args.embeddings_out, facebank_arr)
    np.save(args.labels_out, labels_arr)

    print(f"Saved embeddings: {args.embeddings_out} shape={facebank_arr.shape}")
    print(f"Saved labels: {args.labels_out} shape={labels_arr.shape}")
    print("Flat-folder face bank creation completed.")


if __name__ == "__main__":
    main()
