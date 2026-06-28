import os
import base64
from flask import Flask, request, jsonify, send_from_directory
import requests
from dotenv import load_dotenv

# Load .env locally
load_dotenv()

app = Flask(__name__, static_folder="static")

# Read environment variables
VISION_KEY = os.getenv("VISION_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")

if VISION_ENDPOINT:
    VISION_ENDPOINT = VISION_ENDPOINT.rstrip("/")

print("VISION_KEY exists:", bool(VISION_KEY))
print("VISION_ENDPOINT:", VISION_ENDPOINT)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if not VISION_KEY or not VISION_ENDPOINT:
        return jsonify({
            "error": "Azure Vision not configured",
            "VISION_KEY_EXISTS": bool(VISION_KEY),
            "VISION_ENDPOINT": VISION_ENDPOINT
        }), 500

    data = request.get_json(silent=True) or {}
    image_url = data.get("url")
    image_base64 = data.get("image_base64")

    api_url = (
        f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze"
        f"?api-version=2023-10-01&features=Tags,Read"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": VISION_KEY
    }

    try:
        if image_url:
            headers["Content-Type"] = "application/json"
            resp = requests.post(
                api_url,
                headers=headers,
                json={"url": image_url},
                timeout=20
            )
        elif image_base64:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            image_bytes = base64.b64decode(image_base64)

            headers["Content-Type"] = "application/octet-stream"

            resp = requests.post(
                api_url,
                headers=headers,
                data=image_bytes,
                timeout=20
            )
        else:
            return jsonify({"error": "No image provided"}), 400

    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502

    try:
        return jsonify(resp.json()), resp.status_code
    except Exception:
        return jsonify({"error": resp.text}), resp.status_code


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "VISION_KEY_EXISTS": bool(VISION_KEY),
        "VISION_ENDPOINT": VISION_ENDPOINT
    })


if __name__ == "__main__":
    app.run(debug=True)