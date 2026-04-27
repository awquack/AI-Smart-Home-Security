# face_recognition_module.py – Face encoding, storage and identification

import json
import numpy as np
import database
import config

try:
    import face_recognition as fr
    FACE_RECOG_AVAILABLE = True
    print("[FACE] face_recognition library ready")
except ImportError:
    FACE_RECOG_AVAILABLE = False
    print("[FACE] face_recognition not installed — run: pip install face_recognition")


def encode_face_from_path(image_path):
    """
    Load an image file and return a list of 128-d face encodings found in it.
    Returns [] if no faces detected or library unavailable.
    """
    if not FACE_RECOG_AVAILABLE:
        return []
    image     = fr.load_image_file(image_path)
    encodings = fr.face_encodings(image)
    return encodings


def identify_face(face_encoding, tolerance=None):
    """
    Compare a face encoding against all known faces in the database.
    Returns (name, distance) — name is 'intruder' if no match found.
    """
    if tolerance is None:
        tolerance = config.FACE_TOLERANCE

    known = database.get_known_faces()
    if not known:
        return "unknown", 1.0

    names     = [row["name"] for row in known]
    encodings = [np.array(json.loads(row["encoding"])) for row in known]

    distances = fr.face_distance(encodings, face_encoding)
    min_idx   = int(np.argmin(distances))
    min_dist  = float(distances[min_idx])

    if min_dist <= tolerance:
        return names[min_idx], min_dist
    return "intruder", min_dist


def check_frame_for_faces(frame_bgr, person_box=None):
    """
    Run face recognition on a frame or a cropped person bounding box.

    person_box: (x, y, w, h) from YOLO — crops to that region if provided.

    Returns list of dicts:
        [{"name": str, "distance": float, "is_intruder": bool, "box": (top,right,bottom,left)}]
    """
    if not FACE_RECOG_AVAILABLE or not config.FACE_RECOGNITION_ENABLED:
        return []

    if person_box is not None:
        x, y, w, h = person_box
        pad = int(h * 0.12)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame_bgr.shape[1], x + w + pad)
        y2  = min(frame_bgr.shape[0], y + h + pad)
        crop = frame_bgr[y1:y2, x1:x2]
    else:
        crop = frame_bgr
        x1 = y1 = 0

    if crop.size == 0:
        return []

    # Must be contiguous uint8 RGB for dlib
    rgb            = np.ascontiguousarray(crop[:, :, ::-1], dtype=np.uint8)
    face_locations = fr.face_locations(rgb, model="hog")
    if not face_locations:
        return []

    encodings = fr.face_encodings(rgb, known_face_locations=face_locations, num_jitters=1)

    results = []
    for enc, loc in zip(encodings, face_locations):
        name, dist = identify_face(enc)
        top, right, bottom, left = loc
        results.append({
            "name":        name,
            "distance":    dist,
            "is_intruder": name == "intruder",
            "box":         (left + x1, top + y1, right - left, bottom - top),
        })
    return results
