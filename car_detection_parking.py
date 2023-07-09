import cv2
import imutils
import numpy as np
import pytesseract
import time
import pandas as pd
from gpiozero import AngularServo

servo = AngularServo(18, initial_angle=0, min_pulse_width=0.0006, max_pulse_width=0.0023)

# thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/yashkulkarni/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/yashkulkarni/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/yashkulkarni/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    flag = 0
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    # print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                flag = 1
                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # servo.angle = -90
                    # time.sleep = 2
                    # servo.angle = 90

    return img, objectInfo, flag


def check_string_in_csv(search_string):
    df = pd.read_csv('/home/yashkulkarni/Downloads/Vehicle Number Plate Dataset.csv', header=None)
    return df[0].str.contains(search_string).any()


def capture_noplate():
    img = cv2.imread(r"/home/yashkulkarni/Desktop/Object_Detection_Files/captured_image.jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (620, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    # cv2.imshow("Grey Scale & Bilateral Filter", gray)
    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours == None: return
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)
    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
    cv2.imshow("Cropped number plate : ", Cropped)
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Detected car plate number is:", text)
    return text


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(10,70)

    while True:
        success, img = cap.read()
        cv2.imshow("Output", img)
        result, objectInfo, flag = getObjects(img, 0.45, 0.2, objects=['car', 'bus', 'truck'])
        cv2.imshow("Output", img)
        if flag == 1:
            cv2.imwrite("captured_image.jpg", img)
            text = capture_noplate()
            cv2.waitKey(0)
            found = check_string_in_csv(text.strip())
            if found:
                print("Access Granted")
            else:
                print("Entry not valid")
            cv2.waitKey(0)
        else:
            continue   
