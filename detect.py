"""
detect.py - Real-time Face Mask Detector
Label map: 0 = with_mask, 1 = without_mask
"""
import os, time, urllib.request, pathlib
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ══════════════════════════════════════════
CAMERA_INDEX = 1   # Change if wrong camera
# ══════════════════════════════════════════

# FIXED label map — matches train_model.py exactly
# Index 0 = with_mask (GREEN)
# Index 1 = without_mask (RED)
LABEL_MAP = {
    0: ("Mask",    (0, 200, 80)),   # GREEN
    1: ("No Mask", (0, 60, 220)),   # RED
}

MODEL_PATH     = "mask_detector.keras"
FACE_PROTO     = "face_detector/deploy.prototxt"
FACE_MODEL     = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_MIN = 0.5
IMG_SIZE       = 224

WHITE  = (255, 255, 255)
YELLOW = (0, 200, 255)
GRAY   = (120, 120, 120)


def download_face_detector():
    pathlib.Path("face_detector").mkdir(exist_ok=True)
    files = {
        FACE_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        FACE_MODEL: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    for path, url in files.items():
        if not os.path.exists(path):
            print(f"📥 Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)
    print("✅ Face detector ready!")


def open_camera(index):
    print(f"📷 Opening camera {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    print("   Warming up (2s)...")
    time.sleep(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(15):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print("✅ Camera ready!")
            return cap
        time.sleep(0.1)
    cap.release()
    return None


def draw_rounded_rect(img, x1, y1, x2, y2, color, r=10, t=2):
    cv2.line(img, (x1+r,y1),(x2-r,y1), color, t)
    cv2.line(img, (x1+r,y2),(x2-r,y2), color, t)
    cv2.line(img, (x1,y1+r),(x1,y2-r), color, t)
    cv2.line(img, (x2,y1+r),(x2,y2-r), color, t)
    cv2.ellipse(img,(x1+r,y1+r),(r,r),180, 0,90,color,t)
    cv2.ellipse(img,(x2-r,y1+r),(r,r),270, 0,90,color,t)
    cv2.ellipse(img,(x1+r,y2-r),(r,r), 90, 0,90,color,t)
    cv2.ellipse(img,(x2-r,y2-r),(r,r),  0, 0,90,color,t)


def draw_label(img, text, x, y, bg):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_ = cv2.getTextSize(text, font, 0.65, 2)
    cv2.rectangle(img,(x,y-th-12),(x+tw+12,y), bg,-1)
    cv2.putText(img, text,(x+6,y-6), font, 0.65, WHITE, 2)


def draw_hud(frame, faces, masks, fps):
    h = frame.shape[0]
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(270,115),(15,15,15),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    f = cv2.FONT_HERSHEY_SIMPLEX
    GREEN = (0,200,80); RED = (0,60,220)
    cv2.putText(frame,"NOVA  MASK  DETECTOR",      (10,22), f,0.5,YELLOW,1)
    cv2.putText(frame,f"FPS        : {fps:.1f}",   (10,45), f,0.5,WHITE,1)
    cv2.putText(frame,f"Faces      : {faces}",     (10,65), f,0.5,WHITE,1)
    cv2.putText(frame,f"With Mask  : {masks}",     (10,85), f,0.5,GREEN,1)
    cv2.putText(frame,f"No Mask    : {faces-masks}",(10,105),f,0.5,RED,1)
    cv2.putText(frame,"[Q] Quit  [S] Screenshot",  (10,h-10),f,0.4,GRAY,1)


def main():
    print("=" * 50)
    print("  Nova — Real-Time Face Mask Detector")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found!")
        print("   Run:  python train_model.py\n")
        return

    download_face_detector()

    print("\n🧠 Loading model...")
    model = load_model(MODEL_PATH)
    print(f"   Output shape: {model.output_shape}")
    print(f"   Label map: 0=with_mask(GREEN)  1=without_mask(RED)")

    print("👁️  Loading face detector...")
    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        print(f"\n❌ Camera {CAMERA_INDEX} failed! Try changing CAMERA_INDEX.")
        return

    print("\n✅ Running! Q=quit  S=screenshot\n")
    prev = time.time()
    fails = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            fails += 1
            if fails > 30: break
            continue
        fails = 0

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Detect faces
        blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
        face_net.setInput(blob)
        dets = face_net.forward()

        faces, locs = [], []
        for i in range(dets.shape[2]):
            if dets[0,0,i,2] < CONFIDENCE_MIN: continue
            x1=max(0,int(dets[0,0,i,3]*w)); y1=max(0,int(dets[0,0,i,4]*h))
            x2=min(w-1,int(dets[0,0,i,5]*w)); y2=min(h-1,int(dets[0,0,i,6]*h))
            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
            faces.append(preprocess_input(img_to_array(face_rgb)))
            locs.append((x1,y1,x2,y2))

        preds = []
        if faces:
            preds = model.predict(np.array(faces,dtype="float32"), verbose=0)

        mask_count = 0
        for (x1,y1,x2,y2), pred in zip(locs, preds):
            idx = int(np.argmax(pred))          # 0 or 1
            conf = pred[idx] * 100
            label_text, color = LABEL_MAP[idx]
            display = f"{label_text}  {conf:.0f}%"

            if idx == 0: mask_count += 1        # 0 = with_mask

            draw_rounded_rect(frame, x1,y1,x2,y2, color)
            draw_label(frame, display, x1, y1, color)

        now = time.time()
        draw_hud(frame, len(locs), mask_count, 1.0/(now-prev+1e-6))
        prev = now

        cv2.imshow("Nova — Face Mask Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("👋 Bye!")
            break
        elif key == ord("s"):
            fn = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fn, frame)
            print(f"📸 {fn}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()