import mediapipe as mp
import cv2
import numpy as np
import time
import madiapipe.framework import landmark_pd2
import random


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
score=0

x_enemy=random.randint(50,600)
y_enemy=random.randint(50,400)



def enemy():
    global x_enemy,y_enemy,score
    cv2.circle(image, (x_enemy,y_enemy), 25  ,(0,200,0),5 )
    
    
    
    
    
video = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        _, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        
        
        
        imageHeight, imageWidth, _ = image.shape
        results = hands.process(image)
        
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        color=(255,0,255)
        text=cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
        text=cv2.putText(image,str(score),(590,30)                )
        
        
        