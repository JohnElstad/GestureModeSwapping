#!/usr/bin/env python3
# license removed for brevity

import pyzed.sl as sl
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import rospy
from geometry_msgs.msg import Twist
from collections  import deque

dq = deque(maxlen=5)
linear_vel = 0.0
angular_vel = 0.0
forward_speed = .16
turning_speed = .16

def main():
    # Create a Camera object
    zed = sl.Camera()

    #Set up ros publisher
    pub_move = rospy.Publisher('AGBOT1_cmd_vel', Twist, queue_size=10)
    rospy.init_node('fake_bot_move', anonymous=True)   
    rate = rospy.Rate(10) # 10hz
    cmd = Twist()
    rospy.loginfo("started")

    #import trained model
    model = load_model('/home/user/john_ws/src/gesture_ctrl_john/src/2DBodyModelV3-30E.h5')
    model.summary()

    #mat added 
    runtime = sl.RuntimeParameters()

    mat = sl.Mat()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True
    camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
    str_camera_settings = "BRIGHTNESS"
    step_camera_settings = 1


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:

        exit(1)

    obj_param = sl.ObjectDetectionParameters()
    # Different model can be chosen, optimizing the runtime or the accuracy
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    obj_param.enable_tracking = True
    obj_param.image_sync = True
    obj_param.enable_mask_output = False
    
    # Optimize the person joints position, requires more computations
    obj_param.enable_body_fitting = True

    camera_infos = zed.get_camera_information()
    if obj_param.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        # positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Object Detection: Loading Module...")

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    # For outdoor scene or long range, the confidence should be lowered to avoid missing detections (~20-30)
    # For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
    obj_runtime_param.detection_confidence_threshold = 40
    


    point_data = []
    point_data3D = []



    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        err = zed.retrieve_objects(objects, obj_runtime_param)
        
        key = ''
        if key != 'q':  # for 'q' key
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(mat, sl.VIEW.LEFT)
                frame = mat.get_data()
                # cv2.imshow("ZED", frame)
                key = cv2.waitKey(5)
            else:
                key = cv2.waitKey(5)
                break

        # zed.retrieve_image(mat, sl.VIEW.LEFT)
        # frame = mat.get_data()
        # cv2.imshow("ZED", frame)

        if objects.is_new:
            obj_array = objects.object_list
            print(str(len(obj_array)) + " Person(s) detected\n")
            if len(obj_array) > 0:
                first_object = obj_array[0]
                print("First Person attributes:")
                print(" Confidence (" + str(int(first_object.confidence)) + "/100)")
                if obj_param.enable_tracking:
                    print(" Tracking ID: " + str(int(first_object.id)) + " tracking state: " + repr(
                        first_object.tracking_state) + " / " + repr(first_object.action_state))
                position = first_object.position
                velocity = first_object.velocity
                dimensions = first_object.dimensions
                print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(
                    position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],
                    dimensions[1], dimensions[2]))
                if first_object.mask.is_init():
                    print("2D mask available")

                body_data = []
                body_data3D = []
                print(" Keypoint 2D ")
                keypoint_2d = first_object.keypoint_2d
                for point in keypoint_2d:
                    print("    " + str(point))
                    frame = cv2.circle(frame, (int(point[0]),int(point[1])), 5, (0,0,255), 5)
                    body_data.append(point[0])
                    body_data.append(point[1])

                

                print(body_data)
                print(np.shape(body_data))
                results = model.predict(np.array(body_data).reshape(1,-1))
                confidence = np.max(results)
                prediction = np.argmax(results)
                
                #filtering that takes most common gesture of last 5 frames
                dq.append(prediction)
                fdist = dict(zip(*np.unique(dq,return_counts=True)))
                prediction = list(fdist)[-1]
                #Set the velocity and speed for the rover and text dispalyed on monitor
                outputText = ''

                if prediction == 0:
                    outputText = 'FORWARD'
                    linear_vel = forward_speed
                    angular_vel = 0.0

                elif prediction == 1:
                    outputText = 'LEFT'
                    linear_vel = 0.0
                    angular_vel = -turning_speed

                elif prediction == 2:
                    outputText = 'BACK'
                    linear_vel = -forward_speed
                    angular_vel = 0.0

                elif prediction == 3:
                    outputText = 'RIGHT'
                    linear_vel = 0.0
                    angular_vel = turning_speed

                elif prediction == 4:
                    outputText = 'NO GESTURE DETECTED'
                    linear_vel = 0.0
                    angular_vel = 0.0

                elif prediction == 5:
                    outputText = 'STOP'
                    linear_vel = 0.0
                    angular_vel = 0.0

                #publishes to ROS topic
                cmd.linear.x = linear_vel
                cmd.angular.z = angular_vel
                pub_move.publish(cmd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                rate.sleep()
                  
                frame = cv2.putText(frame, str(outputText), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       2, [255,0,0], 2, cv2.LINE_AA)
                frame = cv2.putText(frame, str(confidence), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, [0,255,0], 2, cv2.LINE_AA)
                
                print(prediction)

                cv2.imshow("ZED", frame)
                
                ##USE IF USING 3D data points
                # print("\n Keypoint 3D ")
                # keypoint = first_object.keypoint
                
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

