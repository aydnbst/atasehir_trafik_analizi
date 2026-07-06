import cv2
from ultralytics import YOLO
import json
import time
import os
from flask import Flask, Response, send_from_directory
from flask_cors import CORS
import threading

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-1"

app = Flask(__name__, root_path=os.path.dirname(os.path.abspath(__file__)))
CORS(app)

output_frame = None
lock = threading.Lock()

# BULUT OPTİMİZASYONU: İşlemciyi yormamak için 'yolov8s' yerine 'yolov8n' (Nano) modeline geçtik
model = YOLO("yolov8n.pt") 
VIDEO_SOURCE = "https://601a43eea2819.streamlock.net/hls/974.stream/chunklist.m3u8"

EMISSION_FACTORS = {2: 0.15, 3: 0.08, 5: 0.60, 7: 0.80}
JSON_PATH = "live_traffic_data.json"

def video_processing_thread():
    global output_frame
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    total_co2_emitted = 0.0
    last_json_update = time.time()
    counted_centers = []
    frame_count = 0  # Kare atlama sayacı

    print("Yapay Zeka Motoru Hafifletilmis Modda Calisiyor...")

    while True:
        if not cap.isOpened():
            time.sleep(2)
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            continue

        success, frame = cap.read()
        if not success:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            continue

        frame_count += 1
        current_vehicle_count = 0

        # BULUT OPTİMİZASYONU: Her karede değil, 3 karede bir YOLO'yu çalıştırarak CPU yükünü düşürüyoruz
        if frame_count % 3 == 0:
            # conf değerini 0.20 yaparak hızı ve doğruluğu dengede tuttuk
            results = model.predict(frame, verbose=False, classes=[2, 3, 5, 7], conf=0.20, iou=0.45)
            
            if results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, cls in zip(boxes, clss):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    current_vehicle_count += 1

                    is_new_vehicle = True
                    for old_cx, old_cy in counted_centers:
                        if ((cx - old_cx)**2 + (cy - old_cy)**2)**0.5 < 35:
                            is_new_vehicle = False
                            break
                    
                    if is_new_vehicle:
                        counted_centers.append((cx, cy))
                        total_co2_emitted += 100 * EMISSION_FACTORS.get(cls, 0.15)

                    label = f"{model.names[cls]}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

            if len(counted_centers) > 500:
                counted_centers = counted_centers[-250:]

            # JSON Çıktısı Üretme
            if time.time() - last_json_update > 2.0:
                last_json_update = time.time()
                status_color = "green" if current_vehicle_count < 4 else "orange" if current_vehicle_count < 9 else "red"
                
                data_to_export = {
                    "status_color": status_color,
                    "current_vehicle_count": int(current_vehicle_count),
                    "total_vehicles_passed": int(len(counted_centers)),
                    "total_co2_grams": round(total_co2_emitted, 2),
                    "last_update": time.strftime("%H:%M:%S")
                }
                with open(JSON_PATH, "w") as f:
                    json.dump(data_to_export, f, indent=4)

        # Görsel bilgilendirmeyi ekle
        cv2.putText(frame, f"Anlik Arac: {current_vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Kareyi web yayını havuzuna kopyala
        with lock:
            output_frame = frame.copy()
            
        # Bulut sunucusunun nefes alması için küçük bir bekleme (Sleep)
        time.sleep(0.01)

    cap.release()

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            suc, encoded_img = cv2.imencode('.jpg', output_frame)
            if not suc:
                continue
            frame_bytes = encoded_img.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # BULUT OPTİMİZASYONU: Yayın hızını ~15 FPS bandına sabitleyerek bant genişliğini koruyoruz
        time.sleep(0.06) 

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/live_traffic_data.json')
def get_json():
    return send_from_directory('.', 'live_traffic_data.json')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=video_processing_thread)
    t.daemon = True
    t.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
