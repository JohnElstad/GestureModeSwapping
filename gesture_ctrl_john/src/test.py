import mediapipe as mp
import cv2

mp_hands = []
    
        #import hands model from mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(in_detection_confidence=0.8, min_tracking_confidence=0.1,max_num_hands = 1)as hands:
        print(1)