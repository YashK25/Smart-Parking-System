import RPi.GPIO as IO
import urllib.request
from time import sleep
from gpiozero import AngularServo
import cv2
import imutils
import numpy as np
import pytesseract

IO.setmode(IO.BCM)
IO.setup(2, IO.IN)  # GPIO 2 -> IR sensor for Entry Gate
IO.setup(3, IO.IN)  # GPIO 3 -> IR sensor for Parking Slot 1
IO.setup(4, IO.IN)  # GPIO 4 -> IR sensor for Parking Slot 2
IO.setup(17, IO.IN)  # GPIO 17 -> IR sensor for Parking Slot 3
IO.setup(27, IO.IN)  # GPIO 27 -> IR sensor for Parking Slot 4
IO.setup(22, IO.IN)  # GPIO 22 -> IR sensor for Exit Gate
# GPIO 18 -> servo motor @ enterance
# GPIO 23 -> servo motor @ exit

totalslots = 4


def check_parking(totalslots=4):
    p1, p2, p3, p4 = 0, 0, 0, 0
    if IO.input(3) == False:
        totalslots -= 1
        p1 = 1
        print("Parking slot 1 occupied")
    else:
        p1 = 0

    if IO.input(4) == False:
        totalslots -= 1
        p2 = 1
        print("Parking slot 2 occupied")
    else:
        p2 = 0

    if IO.input(17) == False:
        totalslots -= 1
        p3 = 1
        print("Parking slot 3 occupied")
    else:
        p3 = 0

    if IO.input(27) == False:
        totalslots -= 1
        p4 = 1
        print("Parking slot 4 occupied")
    else:
        p4 = 0

    f = urllib.request.urlopen(
        "https://api.thingspeak.com/update?api_key=K19NFNWNJ5S8RET8&field1=%s&field2=%s&field3=%s&field4=%s&field5=%s" % (
        p1, p2, p3, p4, totalslots))
    return totalslots


def operate_gate(pin):
    servo = AngularServo(pin, min_pulse_width=0.0006, max_pulse_width=0.0023)
    servo.angle = 0
    sleep(1)
    servo.angle = 90
    sleep(5)
    servo.angle = 20
    sleep(1)


def operate_exit():
    servo1 = AngularServo(23, min_pulse_width=0.0006, max_pulse_width=0.0020)
    servo1.angle = 0
    sleep(1)
    servo1.angle = 90
    sleep(5)
    servo1.angle = 10
    sleep(2)


def capture_noplate():
    img = cv2.imread(r"/home/yashkulkarni/Downloads/car_number_plate2.jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (620, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    # cv2.imshow("Grey Scale & Bilateral Filter", gray)
    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    # cv2.imshow("Cropped number plate : ", Cropped)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Detected car plate number is:", text)


#   Main:

while True:
    check_parking()
    if IO.input(2) == False and totalslots != 0:
        print("Car detected at entrance\n")
        capture_noplate()
        operate_gate(18)
        totalslots = check_parking(totalslots)
    sleep(1)
    if IO.input(22) == False:
        print("Car detected at exit\n")
        # capture_noplate()
        operate_exit()
        totalslots = check_parking(totalslots)

