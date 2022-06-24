#!/usr/bin/env python3
# license removed for brevity



# Gesture Control with Implemented Mode Swapping - John Elstad 6/6/2022
# This code implements gesture control with the ag robot. Contains all different modes and a gesture to swap between them. 
# Publishes a ROS twist message /AGBOT_cmd_vel or something like that to control translational and rotational speed
# Gestures can be found in my capstone report. They control left, right, forward, back and mode swap.


import pyzed.sl as sl
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from collections  import deque
import time

prev_prediction = 0
dq = deque(maxlen=5)
personQueue = deque(maxlen=5)
linear_vel = 0.0
angular_vel = 0.0
angular_speed = .09
linear_speed = .2

def follow(position,setPosition):
    angular_speed = .09
    linear_speed = .2
    x = position[0]
    y = position[1]
    z = position[2]

    xSetPoint = setPosition[0]
    zSetPoint = setPosition[2]
  
    Kp = .1
    #positive x is left looked at front of rover, 0 is directly centered at the camera
    #Positie z is distance from camera. 0 is on camera



    # cmd_vel pub rules 
    x_thresh = .2*z#meters from center of camera           
    follow_distance = zSetPoint
    z_thresh = .5
    if x_thresh < 0.5:
        x_thresh = 0.5


    if z - follow_distance > z_thresh:
        #go forwards
        linear_vel = abs(z-follow_distance)*Kp

    elif z - follow_distance < -z_thresh:
        #Go backwards
        linear_vel = -(abs(z-follow_distance)*Kp)
    else:
        #don't move
        linear_vel = 0.0

    if x > x_thresh + xSetPoint:
        #turn right
        angular_vel = -angular_speed
        linear_vel = 0 
    elif x < -x_thresh + xSetPoint:
        #turn left
        angular_vel = angular_speed
        linear_vel = 0
    else:
        angular_vel = 0.0
        #don't turn
    print(f'z-follow: {z-follow_distance}')
    
    if(abs(linear_vel)>linear_speed):
        linear_vel = linear_speed
    if(abs(angular_vel)>angular_speed):
        angular_vel = angular_speed


    return linear_vel, angular_vel

def main():
    # Create a Camera object
    zed = sl.Camera()

    #Set up ros publisher
    pub_move = rospy.Publisher('AGBOT1_cmd_vel', Twist, queue_size=10)
    pub_mode = rospy.Publisher('AGBOT1_mode_status',Int8,queue_size = 10)
    pub_x = rospy.Publisher('AGBOT1_posx',Float64,queue_size = 10)
    pub_z = rospy.Publisher('AGBOT1_posz',Float64,queue_size = 10)
    pub_x_ref = rospy.Publisher('AGBOT1_posx_ref',Float64,queue_size = 10)
    pub_z_ref = rospy.Publisher('AGBOT1_posz_ref',Float64,queue_size = 10)
    pub_person_pose = rospy.Publisher('person_pose',Pose,queue_size = 10)
    person_pose = Pose()
    


    rospy.init_node('john_gesture_modes', anonymous=True)   

    rate = rospy.Rate(10) # 10hz
    cmd = Twist()
    rospy.loginfo("started")

    #import trained model
    model = load_model('/home/user/john_ws/src/gesture_ctrl_john/src/2DBodyModelV4XGesture.h5')
    model.summary()

    #mat added 
    runtime = sl.RuntimeParameters()

    mat = sl.Mat()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD720 video mode on jetson
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True
    camera_setteds = sl.VIDEO_SETTINGS.BRIGHTNESS
    str_camera_setteds = "BRIGHTNESS"
    step_camera_setteds = 1

    prev_prediction = -1
    mode = 0
    currentID = 0
    tracked_object = None

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:

        exit(1)

    obj_param = sl.ObjectDetectionParameters()
    # Different model can be chosen, optimized the runtime or the accuracy
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    obj_param.enable_tracking = True
    obj_param.image_sync = True
    obj_param.enable_mask_output = False
    
    # Optimize the person joints position, requires more computations
    obj_param.enable_body_fitting = True

    camera_infos = zed.get_camera_information()
    if obj_param.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        # positional_tracked_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Object Detection: Loaded Module...")

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    # For outdoor scene or long range, the confidence should be lowered to avoid missed detections (~20-30)
    # For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
    obj_runtime_param.detection_confidence_threshold = 40
    


    
    setPosition = [0,0,2]
    t0 = time.time()


    while zed.grab() == sl.ERROR_CODE.SUCCESS:# main loop that detects objects then runs gesture recognition model on them
        linear_vel = 0.0
        angular_vel = 0.0
        angular_speed = .3
        linear_speed = .3
        err = zed.retrieve_objects(objects, obj_runtime_param)
        prev_position = np.array([0,0,0])
       
        key = cv2.waitKey(10)
        if key != ord('q'):  # for 'q' key
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(mat, sl.VIEW.LEFT)
                frame = mat.get_data()
                # cv2.imshow("ZED", frame)
                
        else:
            
            break

        # zed.retrieve_image(mat, sl.VIEW.LEFT)
        # frame = mat.get_data()
        # cv2.imshow("ZED", frame)

        if objects.is_new:
            obj_array = objects.object_list
            print(str(len(obj_array)) + " Person(s) detected\n")
            if len(obj_array) > 0:
                
                

                #Person tracking
                if tracked_object is not None:
                    for person in obj_array:
                        if person.id == tracked_object.id:
                            tracked_object = person
                            break
                        else:
                            tracked_object = person
                else:   
                    tracked_object = obj_array[0]

                print("First Person attributes:")
                print(" Confidence (" + str(int(tracked_object.confidence)) + "/100)")

                if obj_param.enable_tracking:
                    print(" Tracked ID: " + str(int(tracked_object.id)) + " tracked state: " + repr(
                        tracked_object.tracking_state) + " / " + repr(tracked_object.action_state))
                position = tracked_object.position
                velocity = tracked_object.velocity
                dimensions = tracked_object.dimensions
                

                body_data = []
                body_data3D = []
                # print(" Keypoint 2D ")

                for person in obj_array:
                    print(person.head_position)
                    frame = cv2.putText(frame, str(person.id), (int(person.keypoint_2d[0][0]),int(person.keypoint_2d[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 5, cv2.LINE_AA)
                    if person == tracked_object:
                        color = (0,255,0)  
                    else:
                        color = (255,0,0)
                    
                    keypoint_2d = person.keypoint_2d
                    for point in keypoint_2d:
                        if person == tracked_object:
                            body_data.append(point[0])
                            body_data.append(point[1])
                            # print("    " + str(point))
                        frame = cv2.circle(frame, (int(point[0]),int(point[1])), 5, color, 5)
                            
                

                # print(body_data)
                # print(np.shape(body_data))
                results = model.predict(np.array(body_data).reshape(1,-1))
                confidence = np.max(results)
                prediction = np.argmax(results)
                
                #filtered that takes most common gesture of last 5 frames
                dq.append(prediction)
                fdist = dict(zip(*np.unique(dq,return_counts=True)))
                prediction = list(fdist)[-1]

                #bool for if the gesture has changed in the last frame
                gestureSwapped = (prediction != prev_prediction)

                #Set the velocity and speed for the rover and text dispalyed on monitor
                outputText = ''

                if prediction == 0:
                    outputText = 'FORWARD'
                    linear_vel = linear_speed
                    angular_vel = 0.0

                elif prediction == 1:
                    outputText = 'LEFT'
                    linear_vel = 0.0
                    angular_vel = -angular_speed

                elif prediction == 2:
                    outputText = 'BACK'
                    linear_vel = -linear_speed
                    angular_vel = 0.0

                elif prediction == 3:
                    outputText = 'RIGHT'
                    linear_vel = 0.0
                    angular_vel = angular_speed

                elif prediction == 5:
                    outputText = 'NO GESTURE DETECTED'
                    linear_vel = 0.0
                    angular_vel = 0.0

                elif prediction == 4:#MODE SWAP
                    x = position[0]
                    z = position[2]
                    if -.5 < x < .5 and 1.5 < z < 5:
                        if gestureSwapped:
                            t0 = time.time()
                        time_on_gesture = time.time() - t0
                        print(f'time_on_gesture: {time_on_gesture}')
                        if time_on_gesture > 2:
                            if mode != 4:
                                mode = mode + 1
                                t0 = time.time()
                            else:
                                mode = 0
                                t0 = time.time()
                        frame = cv2.putText(frame, str(mode), (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, [0,255,0], 5, cv2.LINE_AA)
                        outputText = 'MODE SWAP...'
                    else:
                        frame = cv2.putText(frame, str(mode), (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, [0,0,255], 5, cv2.LINE_AA)
                



                if mode == 0: #Manual Mode: Defaults to gesture movement control
                   pass

                elif mode ==1:  #Follow Me Mode
                    if outputText != 'MODE SWAP...':
                        outputText = 'FOLLOWING...'
                        linear_vel , angular_vel = follow(position,setPosition)
                    if prediction == 3:
                        outputText = 'SETTING FOLLOW DISTANCE...'
                        setPosition = position
                        linear_vel = 0 
                        angular_vel = 0
                    
                else: #if not in mode 1 or 2 don't do anything
                    linear_vel = 0
                    angular_vel = 0

                cmd.linear.x = linear_vel
                cmd.angular.z = angular_vel
                pub_move.publish(cmd)
                pub_mode.publish(mode)

                person_pose.position.x = position[0]
                person_pose.position.y = position[1]
                person_pose.position.z = position[2]

                pub_person_pose.publish(person_pose)

                pub_x.publish(position[0])
                pub_z.publish(position[2])
                pub_x_ref.publish(setPosition[0])
                pub_z_ref.publish(setPosition[2])
                rate.sleep()
                print(setPosition)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                
                
                frame = cv2.putText(frame, str(outputText), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       2, [255,0,0], 2, cv2.LINE_AA)
                frame = cv2.putText(frame, str(confidence), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, [0,255,0], 2, cv2.LINE_AA)
                
                print(prediction)
                prev_prediction = prediction
                cv2.imshow("ZED", frame)
                
                ##USE IF USING 3D data points
                # print("\n Keypoint 3D ")
                # keypoint = tracked_object.keypoint
                
                # for point in keypoint:
                #     body_data3D.append(point[0])
                #     body_data3D.append(point[1])
                #     body_data3D.append(point[2])
                #     print("    " + str(point))

                
                # print(body_data3D)
                # print(np.shape(body_data3D))
                

    # Close the camera
    zed.close()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

