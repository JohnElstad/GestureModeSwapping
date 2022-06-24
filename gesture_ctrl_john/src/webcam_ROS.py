#!/usr/bin/env python3
# license removed for brevity

from geometry_msgs.msg import Twist
import mediapipe as mp
import rospy
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def get_gesture(frame, model, mp_hands, mp_drawing): # Returns Gesture : 0 = Nothing, 1= Hands up, 2 = Thumbs up
    print('Getting Gesture')
    prediction = 0
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5,max_num_hands = 1) as hands:
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,1)
            
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
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
                outputText = ''
                if prediction == 0:
                    outputText = 'No gesture detected'
                elif prediction == 1:
                    outputText = 'STOP'
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
                cv2.imshow('Hand Tracking',image)
                print(prediction)
                
        return prediction

def main():
    #MEDIAPIPE stuff
#    
        mp_hands = []
    
        #import hands model from mediapipe
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        
        #initialize camera
        cap = cv2.VideoCapture(0)
        key = ''

        #Create and load Model
        model = load_model('/home/john/catkin_ws/src/gesture_ctrl_john/src/CapstoneAgModelV5.h5_no0')
        #V3 works okay
        #V4 is meh
        model.summary()
        
        #ROS Stuff
        pub_move = rospy.Publisher('AGBOT1_cmd_vel', Twist, queue_size=10)
        rospy.init_node('fake_bot_move', anonymous=True)
        rate = rospy.Rate(10) # 10hz
        cmd = Twist()
        rospy.loginfo("started")
        
        while not rospy.is_shutdown():
            linear_vel = 0.0
            angular_vel = 0.0
            #get frame from zed camera

            if(cap.isOpened()):
                ret, frame = cap.read()
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()


            gesture = get_gesture(frame, model, mp_hands, mp_drawing)
            if gesture == 0: #No gesture Selected
#                    linear_vel = 0.04
                linear_vel = 0.0
                angular_vel = 0.0
            elif gesture == 1: #Stop
                linear_vel = 0.0
                angular_vel = 0.0
            elif gesture == 2: #Move Forward
                linear_vel = 0.05    	
                angular_vel = 0.0
            elif gesture == 3: #Move left
                linear_vel = 0.0    	
                angular_vel = 0.05
            elif gesture == 4: #Move right
                linear_vel = 0.00    	
                angular_vel = -.05
            elif gesture == 5: #Move back
                linear_vel = -0.05    	
                angular_vel = 0.0

            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            pub_move.publish(cmd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            rate.sleep()


    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


    
