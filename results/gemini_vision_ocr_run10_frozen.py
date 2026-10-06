import json
import os
import time
import warnings
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import APIError

warnings.filterwarnings("ignore", category=UserWarning)

def extract_vrc_data(image_path: str, max_retries: int = 3) -> str:
    client = genai.Client()
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = """
    Extract structured JSON from this document.
    If it is a Ghana Card, include: document_type, pin_number, surname, first_names, date_of_birth, sex, issue_date, expiry_date.
    If it is a Vehicle Registration Certificate (VRC), include: document_type, vehicle_registration_number, chassis_number, make, model, colour, seating_capacity, vehicle_type, private_or_commercial, owner_name.
    Return ONLY valid JSON matching this structure.
    """

    models_to_try = ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    for model_name in models_to_try:
        print(f"\n  🎯 Model selected: {model_name}")
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Attempting extraction with {model_name} (Attempt {attempt}/{max_retries})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )

                clean_data = json.loads(response.text)
                
                out_dir = Path("outputs")
                out_dir.mkdir(exist_ok=True)
                out_filename = out_dir / f"{Path(image_path).stem}_extracted.json"
                
                with open(out_filename, "w") as out_f:
                    json.dump(clean_data, out_f, indent=2)

                print(f"  ✅ Extraction successful via {model_name} -> Saved to {out_filename}")
                return str(out_filename)

            except APIError as e:
                print(f"  ⚠️ API error ({e.code}): {e.message}")
                if attempt < max_retries:
                    sleep_time = attempt * 3
                    print(f"  ⏳ Retrying {model_name} in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"  ⚠️ Exhausted retries for {model_name}. Switching to fallback model...")
            except Exception as e:
                print(f"  ❌ Unexpected error with {model_name}: {e}")
                break

    raise RuntimeError(f"Failed to process {image_path} after trying all available models: {models_to_try}")
