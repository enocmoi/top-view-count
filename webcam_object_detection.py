



#%% Packages imports
import cv2
import time
import json
import threading
import http.client
import numpy as np
from yolov7 import YOLOv7

def retrieve_json(count):
    dictionary = {
    "count": count,
    }
    with open("../results/sample.json", "w") as outfile:
        json.dump(dictionary, outfile)

    return dictionary


def send_bin(dictionary):

    conn = http.client.HTTPSConnection('enq2n0q1b90d.x.pipedream.net')
    conn.request("POST", "/", f'{dictionary}', {'Content-Type': 'application/json'})


def process_boxes(boxes, scores, class_ids, pts):
    iter_boxes = 0
    for i in range(len(boxes)):
        box, score, class_id = boxes[i], scores[i], class_ids[i]
        confidence = score
        xCenter = (box[0] + box[2]) / 2
        yCenter = box[3]
        if class_id == 0 and confidence > 0.5 and 200 < xCenter < 1000 and 500 < yCenter < 1000:
            point = (xCenter, yCenter)
            if cv2.pointPolygonTest(pts, point, False) > 0:
                iter_boxes += 1

    return iter_boxes


def main():
    # Initialize the webcam
    cap = cv2.VideoCapture("rtsp://admin:Abcd1234!@172.16.2.7:8554")

    # Initialize YOLOv7 object detector
    model_path = "models/yolov7-tiny_480x640.onnx"
    yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

    # Choose fps
    frame_rate = 10
    prev = 0

    # Modify window's name
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    print(frame_rate)
    # Polygon
    pts = np.array([[417, 519],[1217, 759],[537, 1067],[9, 671]], np.int32)
    pts = pts.reshape((-1,1,2))
    # Find middle of polygon
    milieu = np.mean(pts, axis=0)
    milieu_x, milieu_y = milieu[0][0], milieu[0][1]
    iter_box_then = 0
    while True:
        time_elapsed = time.time() - prev
        ret, frame = cap.read()

        if time_elapsed > 1./frame_rate:
            prev = time.time()

            # if not ret:
            #     break
            
            cv2.polylines(frame,[pts],True,(0,255,255), 3)
            # Update object localizer
            boxes, scores, class_ids = yolov7_detector(frame)
            iter_boxes = process_boxes(boxes, scores, class_ids, pts)
            if iter_boxes!=iter_box_then:
                dictionary = retrieve_json(iter_boxes)
                send_bin_thread = threading.Thread(target=send_bin, args=(dictionary,))
                send_bin_thread.start()
                iter_box_then = iter_boxes

            # Définissez la position et la police de texte
            position = (round(milieu_x), round(milieu_y))  
            police = cv2.FONT_HERSHEY_SIMPLEX  
            cv2.putText(frame, str(iter_boxes), position, police, 2, (0,255,255), 2)

            combined_img = yolov7_detector.draw_detections(frame)
            cv2.imshow("Detected Objects", combined_img)

            # Press key q to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

main()
