import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps


def load_and_deskew_image(image_path: Path) -> np.ndarray:
    """
    Loads an image, respects EXIF orientation, and normalizes standard dimensions.
    """
    pil_img = Image.open(image_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Standardize image width to 1600px maintaining aspect ratio
    h, w = cv_img.shape[:2]
    target_width = 1600
    target_height = int(target_width * (h / w))
    return cv2.resize(cv_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def auto_rotate_to_horizontal(image: np.ndarray) -> np.ndarray:
    """
    Evaluates 4 rotations (0, 90, 180, 270) and selects the orientation with the 
    highest keyword match score.
    """
    anchor_keywords = ["GHANA", "VEHICLE", "REGISTRATION", "CHASSIS", "MAKE", "MODEL", "COLOUR", "OWNER"]
    best_angle = 0
    max_score = -1
    best_img = image

    angles = [0, 90, 180, 270]
    for angle in angles:
        if angle == 0:
            rotated = image
        elif angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(gray, config="--psm 11 --oem 1").upper()
        score = sum(1 for kw in anchor_keywords if kw in txt)

        if score > max_score:
            max_score = score
            best_angle = angle
            best_img = rotated

    return best_img


def preprocess_vrc_image(image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE and Otsu's adaptive thresholding to remove security watermarks
    and enhance text lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Otsu thresholding for clean binarization
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_word_boxes(processed_img: np.ndarray) -> list:
    """
    Extracts word bounding boxes and confidence levels using Tesseract PSM 6.
    """
    data = pytesseract.image_to_data(processed_img, output_type=Output.DICT, config="--oem 1 --psm 6")
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip().upper()
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if text and conf > 15:
            words.append({
                "text": text,
                "left": data["left"][i],
                "top": data["top"][i],
                "right": data["left"][i] + data["width"][i],
                "bottom": data["top"][i] + data["height"][i]
            })
    return words


def fuzzy_match(target: str, candidate: str, threshold: float = 0.7) -> bool:
    return SequenceMatcher(None, target, candidate).ratio() >= threshold


def find_field_by_spatial_anchor(words: list, label_keywords: list, max_x_dist: int = 450, max_y_diff: int = 35) -> str:
    """
    Locates text fields positioned relative to key label anchors (Right or Directly Below).
    """
    stop_words = {"OF", "VEHICLE", "COLOUR", "COLOR", "MODEL", "TYPE", "USE", "BODY", "REG", "NO", "NAME", "OWNER"}

    for word in words:
        for keyword in label_keywords:
            if fuzzy_match(keyword, word["text"]):
                anchor_right = word["right"]
                anchor_top = word["top"]

                candidates = []
                for w in words:
                    x_dist = w["left"] - anchor_right
                    y_diff = abs(w["top"] - anchor_top)
                    
                    # Capture words horizontally aligned to the right of the anchor label
                    if 5 <= x_dist <= max_x_dist and y_diff <= max_y_diff:
                        candidates.append((w["left"], w["text"]))

                candidates.sort(key=lambda x: x[0])
                val_words = [w[1] for w in candidates if w[1] not in stop_words]
                cleaned = re.sub(r"[^A-Z0-9\s-]", "", " ".join(val_words)).strip()

                if len(cleaned) >= 2:
                    return cleaned
    return None


def clean_chassis_vin(vin_str: str) -> str:
    """
    Sanitizes extracted Chassis/VIN strings according to ISO 3779 rules.
    """
    if not vin_str:
        return None
    cleaned = re.sub(r"[^A-Z0-9]", "", vin_str.upper())
    # Standard VINs do not use I, O, Q to avoid OCR confusion with 1, 0
    cleaned = cleaned.replace("I", "1").replace("O", "0").replace("Q", "0")
    return cleaned if len(cleaned) >= 8 else None


def parse_vrc_pipeline(image_path: Path) -> dict:
    raw_img = load_and_deskew_image(image_path)
    oriented_img = auto_rotate_to_horizontal(raw_img)
    processed = preprocess_vrc_image(oriented_img)

    words = extract_word_boxes(processed)
    full_str = " ".join([w["text"] for w in words])

    # Ghanaian Registration Regex: GX 1234-21, GS 1234-20, GT 9999 X, GE 123-19
    reg_match = re.search(r"\b([A-Z]{2}\s*[-]?\s*\d{1,4}\s*[-]?\s*(\d{2}|[A-Z]))\b", full_str)

    # VIN / Chassis Search
    vin_match = re.search(r"\b([1-9A-HJ-NPR-Z0-9]{10,17})\b", full_str)

    # Spatial Parsing with Dynamic Anchors
    owner = find_field_by_spatial_anchor(words, ["OWNER", "NAME", "REGISTERED"])
    make = find_field_by_spatial_anchor(words, ["MAKE", "MANUFACTURER"])
    model = find_field_by_spatial_anchor(words, ["MODEL", "VARIANT"])
    colour = find_field_by_spatial_anchor(words, ["COLOUR", "COLOR"])
    body = find_field_by_spatial_anchor(words, ["BODY", "TYPE", "LIFTBACK", "SALOON"])

    raw_vin = vin_match.group(1) if vin_match else find_field_by_spatial_anchor(words, ["CHASSIS", "VIN", "FRAME"])

    is_card = "card" in image_path.name.lower() or "CARD" in full_str

    return {
        "document_type": "vrc_card" if is_card else "vrc_book",
        "registered_owner": owner,
        "vehicle_registration_number": reg_match.group(1).replace(" ", "") if reg_match else None,
        "chassis_number": clean_chassis_vin(raw_vin),
        "make": make,
        "model": model,
        "colour": colour,
        "vehicle_type": body,
        "private_or_commercial": "COMMERCIAL" if "COMMERCIAL" in full_str else "PRIVATE",
        "source_file": image_path.name
    }


def extract_vrc(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    data = parse_vrc_pipeline(path)

    required_fields = [
        "registered_owner", "vehicle_registration_number", "chassis_number",
        "make", "model", "colour", "vehicle_type"
    ]
    data["review_required_fields"] = [f for f in required_fields if data[f] is None]

    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/vrc_ocr.py <image_path>")
        sys.exit(1)

    try:
        result = extract_vrc(sys.argv[1])
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)