import re
from pathlib import Path

from PIL import Image, ImageOps
import pytesseract


MAX_DIMENSION = 2048
TESSERACT_CONFIG = "--oem 1 --psm 3"

GHANA_PIN_PATTERN = re.compile(
    r"\bGHA\s*-\s*(\d(?:\s*\d){8})\s*-\s*(\d)\b",
    re.IGNORECASE,
)

FIRST_NAMES_LABEL_PATTERN = re.compile(
    r"FIRSTNAMES?|PR[ÉE]NOMS",
    re.IGNORECASE,
)

PREVIOUS_NAMES_LABEL_PATTERN = re.compile(
    r"PREVIOUS\s*NAMES?|NOMS?\s*PR[ÉE]C[ÉE]DENTS?",
    re.IGNORECASE,
)


def preprocess_image(image_path: str | Path) -> Image.Image:
    """Open, orient, grayscale, and resize a document image for OCR."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")

        width, height = image.size
        largest_side = max(width, height)

        if largest_side > MAX_DIMENSION:
            scale = MAX_DIMENSION / largest_side
            new_size = (
                round(width * scale),
                round(height * scale),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image.copy()


def extract_ocr_text(processed_image: Image.Image) -> str:
    """Extract raw English OCR text from a processed Ghana Card image."""
    return pytesseract.image_to_string(
        processed_image,
        lang="eng",
        config=TESSERACT_CONFIG,
    )


def clean_name(value: str) -> str | None:
    """Normalize a name line returned by OCR."""
    value = re.sub(r"[^A-ZÀ-ÖØ-Ý' -]", " ", value.upper())
    value = re.sub(r"\s+", " ", value).strip()

    return value if len(value) >= 2 else None


def extract_ghana_card_number(ocr_text: str) -> str | None:
    """Find and normalize a Ghana Card number into GHA-#########-#."""
    match = GHANA_PIN_PATTERN.search(ocr_text)

    if match is None:
        return None

    middle_digits = re.sub(r"\s+", "", match.group(1))
    final_digit = match.group(2)

    return f"GHA-{middle_digits}-{final_digit}"


def extract_client_name(ocr_text: str) -> str | None:
    """
    Extract the cardholder's full name.

    On the Ghana Card OCR layout, surname normally appears immediately
    before the Firstname(s)/Prénoms line, and first names appear after it.
    """
    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    surname = None
    first_names = None

    for index, line in enumerate(lines):
        if FIRST_NAMES_LABEL_PATTERN.search(line):
            if index > 0:
                surname = clean_name(lines[index - 1])

            for next_line in lines[index + 1:]:
                if PREVIOUS_NAMES_LABEL_PATTERN.search(next_line):
                    break

                candidate = clean_name(next_line)

                if candidate:
                    first_names = candidate
                    break

            break

    if surname and first_names:
        return f"{surname} {first_names}"

    return surname or first_names


def run_ocr_pipeline(image_path: str | Path) -> dict:
    """
    Extract insurance-relevant identity data from a Ghana Card image.

    Output is intentionally limited to client_name and ghana_card_number.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    processed_image = preprocess_image(image_path)
    raw_text = extract_ocr_text(processed_image)

    return {
        "image_path": str(image_path),
        "image_width": processed_image.width,
        "image_height": processed_image.height,
        "client_name": extract_client_name(raw_text),
        "ghana_card_number": extract_ghana_card_number(raw_text),
        "raw_text": raw_text,
    }


if __name__ == "__main__":
    result = run_ocr_pipeline(
        "data/sample_documents/ghana_card_sample.jpg"
    )

    print("\n--- GHANA CARD EXTRACTION ---\n")
    print("Client name:", result["client_name"])
    print("Ghana Card number:", result["ghana_card_number"])

    print("\n--- RAW OCR TEXT ---\n")
    print(result["raw_text"])