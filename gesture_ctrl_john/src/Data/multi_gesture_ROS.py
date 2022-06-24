#!/usr/bin/env python3
# license removed for brevity

from geometry_msgs.msg import Twist
import mediapipe as mp
import rospy
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyzed.sl as sl
import time


camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

t0 = time.clock()

def get_gesture(frame, model, mp_hands, mp_drawing): # Returns Gesture : 0 = Nothing, 1= Hands up, 2 = Thumbs up
    print('Getting Gesture')
    prediction = 0
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.1,max_num_hands = 1)as hands:
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,1)
            
        image.flags.writeable = False
        t0 = time.clock()
        results = hands.process(image)
        t1 = time.clock()
        print("time for mediapipe= "+ str(t1-t0))

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

                t0 = time.clock()
                results = model.predict(np.array(flat_list).reshape(1,-1))
                t1 = time.clock()
                print("time for gesture= "+ str(t1-t0))
                confidence = np.max(results)
                prediction = np.argmax(results)


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
                image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
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
        
        #initialize zed camera
        print("Running...")
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD2K	

        init_params.camera_fps = 15
        cam = sl.Camera()
        if not cam.is_opened():
            print("Opening ZED Camera...")
        status = cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()
        print_camera_information(cam)
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

            if key != 113:  # for 'q' key
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat, sl.VIEW.LEFT)
                    frame = mat.get_data()
                    key = cv2.waitKey(5)
                else:
                    key = cv2.waitKey(5)
            else:
                cv2.destroyAllWindows()
                cam.close()
                print("\nFINISH")

            
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
                angular_vel = -0.09
            elif gesture == 4: #Move right
                linear_vel = 0.0    	
                angular_vel = .09
            elif gesture == 5: #Move back
                linear_vel = -0.05    	
                angular_vel = 0.0

            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            pub_move.publish(cmd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            rate.sleep()

def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
    print("Camera FPS: {0}.".format(cam.get_camera_information().camera_fps))
    print("Firmware: {0}.".format(cam.get_camera_information().camera_firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))

    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


    
