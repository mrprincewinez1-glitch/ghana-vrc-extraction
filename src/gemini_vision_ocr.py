import os
import json
from google import genai
from google.genai import types

def extract_vrc_data(image_path):
    """
    Extracts vehicle registration certificate (VRC) data using Gemini Vision OCR.
    Uses gemini-3.6-flash baseline evaluation configuration.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = """
    Extract the following fields from this Ghanaian Vehicle Registration Certificate (VRC) image:
    1. vehicle_registration_number
    2. chassis_number
    3. make
    4. model
    5. colour
    6. seating_capacity
    7. vehicle_type
    8. private_or_commercial

    Return ONLY a valid JSON object with these exact keys. Do not include markdown formatting or backticks.
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt
        ]
    )

    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"raw_text": text, "error": "Failed to parse JSON response"}
