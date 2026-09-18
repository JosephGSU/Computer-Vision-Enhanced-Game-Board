import base64
import os
import cv2
import requests

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ENDPOINT = "https://serverless.roboflow.com/catan-hex-resource-identifier/3"


def classify_hex_crop(crop_image) -> dict:
    if not ROBOFLOW_API_KEY:
        raise ValueError("ROBOFLOW_API_KEY environment variable is not set.")

    # Encode NumPy image array to Base64 JPEG bytes in memory
    _, buffer = cv2.imencode(".jpg", crop_image)
    image_b64 = base64.b64encode(buffer).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {ROBOFLOW_API_KEY}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    response = requests.post(MODEL_ENDPOINT, data=image_b64, headers=headers)
    response.raise_for_status()
    return response.json()


def process_crop_directory(dir_path):
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory '{dir_path}' does not exist.")

    detected_pieces = []
    piece_id = 1

    # Filter for valid image extension types
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    file_list = sorted([
        f for f in os.listdir(dir_path) if f.lower().endswith(valid_extensions)
    ])

    for filename in file_list:
        file_path = os.path.join(dir_path, filename)
        
        # Load image into NumPy array for OpenCV
        img = cv2.imread(file_path)
        if img is None:
            print(f"Skipping unloadable file: {filename}")
            continue

        try:
            # Query Roboflow Model
            prediction = classify_hex_crop(img)
            top_class = prediction.get("top", "unknown")
            confidence = prediction.get("confidence", 0.0)

            detected_pieces.append(
                {
                    "id": piece_id,
                    "filename": filename,
                    "type": f"hex_{top_class}",
                    "confidence": round(float(confidence), 4),
                }
            )
            print(f"[{piece_id}/{len(file_list)}] Processed {filename} -> {top_class} ({confidence:.2%})")
            piece_id += 1

        except requests.exceptions.RequestException as e:
            print(f"API Error processing {filename}: {e}")

    return {
        "timestamp": int(os.times().elapsed),
        "total_detected": len(detected_pieces),
        "pieces": detected_pieces,
    }


if __name__ == "__main__":
    board_data = process_crop_directory("./img_dir/")
    print("\nFinal Output JSON Payload:")
    print(board_data)
