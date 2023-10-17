from flask import Flask, render_template, Response
import cv2
import numpy as np
from HandTracking import HandDetector

app = Flask(__name__)
video_stream = cv2.VideoCapture(0)  # 0 for the default webcam
# Width
video_stream.set(3, 1280)
# Height
video_stream.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)


@app.route('/')
def index():
    return render_template('index.html')


def virtual_drag_drop():
    colorR = (255, 0, 255)

    cx, cy, w, h = 100, 100, 200, 200

    class DragRect():
        def __init__(self, posCenter, size=[200, 200], colorR=(255, 0, 255)):
            self.posCenter = posCenter
            self.size = size
            self.colorR = colorR

        def update(self, cursor):
            cx, cy = self.posCenter
            w, h = self.size
            if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
                self.colorR = (0, 255, 0)
                self.posCenter[0], self.posCenter[1] = cursor[0], cursor[1]
            else:
                self.colorR = (255, 0, 255)

    rectList = []
    for x in range(3):
        rectList.append(DragRect([x * 250 + 150, 150]))

    while True:
        success, img = video_stream.read()  # Use the global video_stream

        if not success or img is None:
            continue

        img = cv2.flip(img, 1)

        allHands, img = detector.findHands(img)
        lmList = []
        if allHands:
            lmList = allHands[0]["lmList"]

        if lmList:
            l, _, _ = detector.findDistance(
                lmList[8], lmList[12], img, draw=False)
            if l < 50:
                cursor = lmList[8]
                for rect in rectList:
                    rect.update(cursor)

        imgNew = np.zeros_like(img, np.uint8)
        for rect in rectList:
            cx, cy = rect.posCenter
            w, h = rect.size
            cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                          (cx + w // 2, cy + h // 2), rect.colorR, cv2.FILLED)

        out = img.copy()
        alpha = 0.0
        mask = imgNew.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

        _, buffer = cv2.imencode('.jpg', out)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/drag_drop')
def drag_drop():
    return Response(virtual_drag_drop(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
