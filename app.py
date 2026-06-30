
import os
import cv2
import base64
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

VISION_KEY = os.getenv("VISION_KEY", "")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT", "").rstrip("/")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

FACE_CASCADE = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "vision": bool(VISION_KEY and VISION_ENDPOINT),
        "google": bool(GOOGLE_API_KEY),
        "opencv": not FACE_CASCADE.empty()
    })

def analyze_image(image_bytes):
    url = (
        f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze"
        "?api-version=2023-10-01"
        "&features=Tags,Read,Caption,Objects,People"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/octet-stream"
    }
    r = requests.post(url, headers=headers, data=image_bytes, timeout=30)
    return r.json()

def detect_faces(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    result = []
    for (x,y,w,h) in faces:
        result.append({
            "rectangle":{
                "left":int(x),
                "top":int(y),
                "width":int(w),
                "height":int(h)
            }
        })
    print(f"OpenCV Faces Detected: {len(result)}")
    return result

def detect_brands(image_base64):
    if not GOOGLE_API_KEY:
        return {}
    url=f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    body={
      "requests":[{
        "image":{"content":image_base64},
        "features":[{"type":"LOGO_DETECTION","maxResults":20}]
      }]
    }
    return requests.post(url,json=body,timeout=30).json()

@app.route("/analyze", methods=["POST"])
def analyze():
    data=request.get_json(silent=True) or {}
    image_url=data.get("url")
    image_base64=data.get("image_base64")

    if not image_url and not image_base64:
        return jsonify({"error":"No image provided"}),400

    if image_url:
        resp=requests.get(image_url,timeout=30)
        image_bytes=resp.content
        image_base64=base64.b64encode(image_bytes).decode()
    else:
        if "," in image_base64:
            image_base64=image_base64.split(",",1)[1]
        image_bytes=base64.b64decode(image_base64)

    vision=analyze_image(image_bytes)
    faces=detect_faces(image_bytes)
    brands=detect_brands(image_base64)

    logos=[]
    if isinstance(brands,dict):
        for r in brands.get("responses",[]):
            for l in r.get("logoAnnotations",[]):
                logos.append({"brand":l.get("description"),"confidence":l.get("score")})

    return jsonify({
        "caption":vision.get("captionResult",{}).get("text",""),
        "tags":vision.get("tagsResult",{}).get("values",[]),
        "objects":vision.get("objectsResult",{}).get("values",[]),
        "people":vision.get("peopleResult",{}).get("values",[]),
        "ocr":vision.get("readResult",{}).get("blocks",[]),
        "faces":faces,
        "brands":logos
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
