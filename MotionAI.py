import numpy as np
import json
import time
import pyautogui as pyautogui
import cv2
import mediapipe as mp
import math
# https://github.com/cvzone/cvzone
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PoseModule import PoseDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FPS import FPS
from roboflow import Roboflow
import os
import requests
import base64
from dotenv import load_dotenv

# Load key-value pairs from .env file
load_dotenv()
class MotionAI():
    def __init__(self, camera, printWrapper)->None:
        self.printWrapper = printWrapper
        self.camera = camera
        self.detector = HandDetector(maxHands=2)
        # self.handDetector = HandDetector(maxHands=2)
        # self.faceDetector = FaceDetector()
        
    def run(self):
        detector = self.detector
        cap = self.camera
        offset = 20
        imgSize = 300
        folder = "Data/C"
        counter = 0
        
        while True:
            success, img = cap.read()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
        
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        
                imgCropShape = imgCrop.shape
        
                aspectRatio = h / w
        
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
        
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
        
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
        
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == ord("s"):
                counter += 1
                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
                print(counter)
    def detectStarterKit(self, img=None):
        # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
        ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL")
        ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION")
        ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
        ROBOFLOW_SIZE = 416
        FRAMERATE = 24
        BUFFER = 0.5
        upload_url = "".join([
            "https://detect.roboflow.com/",
            ROBOFLOW_MODEL,
            "/",
            ROBOFLOW_VERSION,
            "?api_key=",
            ROBOFLOW_API_KEY,
            "&format=image",
            "&stroke=8",
            "&confidence=30"
        ])
        # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
        height, width, channels = img.shape
        scale = ROBOFLOW_SIZE / max(height, width)
        img = cv2.resize(img, (round(scale * width), round(scale * height)))

        # Encode image to base64 string
        retval, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer)

        # Get prediction from Roboflow Infer API
        resp = requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        }, stream=True).raw
        
        # Parse result image
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image, None, None 
    def detectHands(self, img=None, offset = 20, imgSize = 300, showHands = False, folder = "Data/C", isFlip = False):
        detector = self.detector
        
        if img is None:
            return None, None, None

        if isFlip:
            img = cv2.flip(img, 1)
        
        hands, img = detector.findHands(img, flipType=not isFlip)
        
        if not showHands: 
            if isFlip: 
                img = cv2.flip(img, 1)
            return img, None, None
        
        counter = 0
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            return img, imgCrop, imgWhite
        return img, None, None


                    
if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    ai = MotionAI(camera,None)
    ai.run()