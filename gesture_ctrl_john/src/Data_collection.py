import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd

#This code is used to collect gesture data. The output is all a csv file where each line has 64 entries. The first 63 floats are the x,y and z coordinates of all 21
#points mediapipe outputs. The 64th piece of data is the classification. 1 = stop, 2 = forward, 3 = left, 4 = right, 5 = back. 0 is no significant gesture detected

#import hands model from mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

finishedData = []

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5,max_num_hands = 1) as hands:     
    cap = cv2.VideoCapture(0)
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while cap.isOpened():
        ret,frame =cap.read()
        
        #detections
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,1)
            
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        gestureSelection = 0; #0 = no gestures selected, 1 = Hands Up, 2 = Thumbs Up
        flat_list = []
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                coordinateList = []
                k = cv2.waitKey(10)
                if(k == ord('h')): #Hand up - STOP
                    gestureSelection = 1
                    print(gestureSelection)
                
                elif(k == ord('w')): #Thumbs up - MOVE FORWARD
                    gestureSelection = 2
                    print(gestureSelection)
                elif(k == ord('a')): #Thumbs left - MOVE LEFT
                    gestureSelection = 3
                    print(gestureSelection)
                elif(k == ord('d')): #Thumbs right - MOVE RIGHT
                    gestureSelection = 4
                    print(gestureSelection)
                elif(k == ord('s')): #Thumbs down - MOVE DOWN
                    gestureSelection = 5
                    print(gestureSelection)
                else:
                    gestureSelection = 0
                    
                for count,landmark in enumerate(hand.landmark):
                    
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    coordinates = [x,y,z]
                    coordinateList.append(coordinates)
                    
                flat_list = list(np.concatenate(coordinateList).flat)
                flat_list.append(gestureSelection)
            finishedData.append(flat_list)
#                     print(f"X={x}")
#                     print(f"Y={y}")
#                     print(f"Y={y}")
#                     image = cv2.circle(image, [int(x*frameWidth),int(y*frameHeight)], 15, [0,0,255], 5)
                
        cv2.imshow('Hand Tracking',image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(len(finishedData))
    #outputs the data and saves it as CapstoneAgHandsTrainingData.csv
    np.savetxt("CapstoneAgHandsTrainingData.csv", 
           finishedData,
           delimiter =", ", 
           fmt ='% s')