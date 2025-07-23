from flask import Flask, Response
import cv2

app = Flask(__name__)
# Replace with your actual RTSP stream URL from Airtel network
camera_url = "rtsp://admin:Tinyprismlabs@192.168.1.6:554/cam/realmonitor?channel=1&subtype=0"

@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(camera_url)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
