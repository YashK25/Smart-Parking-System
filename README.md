# Smart-Parking-System
The idea of this project is to keep track of the parking slots occupied and keep track of the number plates of the vehicles.

It uses computer vision techniques to detect vehicles and automate the parking process. 
This system aims to provide a more efficient and convenient parking experience by automating the entry and exit procedures.

Object Detection Code (car_detection_parking.py)
The Object Detection code is responsible for detecting vehicles in real-time using a camera. It utilizes the SSD MobileNet V3 model for accurate and fast object detection. When a vehicle is detected, the code captures an image and extracts the number plate using image processing techniques. It then checks the extracted number plate against a pre-existing dataset of valid number plates to determine if the vehicle has access to the parking facility.

Dependencies
The code has the following dependencies:

OpenCV (cv2): Library for computer vision tasks.
Imutils: Library for convenient image processing functions.
NumPy: Library for numerical computations.
PyTesseract: Library for optical character recognition (OCR).
Pandas: Library for data manipulation and analysis.
GPIOZero: Library for controlling GPIO pins (specifically, the AngularServo class).
Setup and Configuration
Before running the code, make sure to:

Install the required dependencies.
Connect a camera to your system or update the code to use the desired video source.
Set the appropriate file paths for the COCO class names, model configuration, and pre-trained weights.

Usage
To use the Object Detection code, follow these steps:

Run the script object_detection.py.
The code will initialize the object detection model and start capturing video frames.
When a vehicle is detected, the system captures an image of the vehicle and extracts the number plate.
The extracted number plate is checked against a CSV dataset to verify access.
If the number plate is found in the dataset, "Access Granted" is displayed. Otherwise, "Entry not valid" is displayed.



Parking System Code (parking-slot.py)
The Parking System code utilizes IR sensors and servo motors to automate the entry and exit procedures of the parking facility. It monitors the occupancy of parking slots using IR sensors and controls servo motors to open/close the entry and exit barriers accordingly. The system also updates the parking status on the ThingSpeak cloud IoT platform to reflect the occupancy of each parking slot.

Integration with ThingSpeak
The Smart Car Parking System is integrated with the ThingSpeak cloud IoT platform for real-time monitoring and data visualization. The system updates the parking status (occupied or vacant) of each parking slot to the ThingSpeak platform using HTTP requests. This enables users to remotely monitor the availability of parking slots and receive notifications when slots become available.

Dependencies
The code has the following dependencies:

RPi.GPIO: Library for accessing GPIO pins on the Raspberry Pi.
urllib.request: Library for making HTTP requests.
gpiozero: Library for controlling GPIO pins (specifically, the AngularServo class).
cv2: OpenCV library for computer vision tasks.
imutils: A library for convenient image processing functions.
numpy: A library for numerical computations.
pytesseract: A library for optical character recognition (OCR).
Setup and Configuration
Before running the code, make sure to:

Connect the IR sensors to the appropriate GPIO pins on the Raspberry Pi.
Connect the servo motors to the desired GPIO pins on the Raspberry Pi.
Configure the servo motor angles and timings in the code.
Usage
To use the Parking System code, follow these steps:

Run the script parking_system.py.
The code will continuously monitor the IR sensors to detect the presence of vehicles at the entrance and exit.
When a vehicle is detected at the entrance, the system captures the number plate and opens the entry barrier.
When a vehicle is detected at the exit, the system opens the exit barrier.
The code updates the parking status on the ThingSpeak platform to reflect the occupancy of each parking slot.
