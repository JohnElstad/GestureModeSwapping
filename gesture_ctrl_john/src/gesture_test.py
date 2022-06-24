import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
# was hoping to make this the gesture recognition model with all gestures that didn't publish to ros.

#import hands model from mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

finishedData = []

model = load_model('CapstoneAgModelV5.h5')

model.summary()

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
                for count,landmark in enumerate(hand.landmark):
                    
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    coordinates = [x,y,z]
                    coordinateList.append(coordinates)
                    
                flat_list = list(np.concatenate(coordinateList).flat)
                confidence = np.max(model.predict(np.array(flat_list).reshape(1,-1)))
                prediction = np.argmax(model.predict(np.array(flat_list).reshape(1,-1)))
                if prediction == 0:
                    outputText = 'Nothing'
                elif prediction == 1:
                    outputText = 'Stop'
                elif prediction == 2:
                    outputText = 'FORWARD'
                elif prediction == 3:
                    outputText = 'LEFT'
                elif prediction == 4:
                    outputText = 'RIGHT'
                elif prediction == 5:
                    outputText = 'BACK'   

                image = cv2.putText(image, str(outputText), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       2, [255,0,0], 2, cv2.LINE_AA)
                image = cv2.putText(image, str(confidence), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, [0,255,0], 2, cv2.LINE_AA)



            
#                     print(f"X={x}")
#                     print(f"Y={y}")
#                     print(f"Y={y}")
#                     image = cv2.circle(image, [int(x*frameWidth),int(y*frameHeight)], 15, [0,0,255], 5)
                
        cv2.imshow('Hand Tracking',image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()