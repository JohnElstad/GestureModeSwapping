"""
Data Collecjtion for Body Model

Data Collection for body tracking. Recordes gestures and outputs the pose positions and gesture labels into a csv file.

"""


from traceback import FrameSummary
from tracemalloc import Frame
import pyzed.sl as sl
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt

def main():


    
    # Create a Camera object
    zed = sl.Camera()

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
    key = []

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
        # positional_tracking_param.set_as_static = TrueP
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


    t0 = time.clock()
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        err = zed.retrieve_objects(objects, obj_runtime_param)
        
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
                first_object = obj_array[0]
                # print("First Person attributes:")
                # print(" Confidence (" + str(int(first_object.confidence)) + "/100)")
                # if obj_param.enable_tracking:
                    # print(" Tracking ID: " + str(int(first_object.id)) + " tracking state: " + repr(
                        # first_object.tracking_state) + " / " + repr(first_object.action_state))
                position = first_object.position
                velocity = first_object.velocity
                dimensions = first_object.dimensions
                # print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D diFmentions: [{6},{7},{8}]".format(
                    # position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],
                    # dimensions[1], dimensions[2]))
                # if first_object.mask.is_init():
                    # print("2D mask available")

                body_data = []
                body_data3D = []
                # print(" Keypoint 2D ")
                keypoint_2d = first_object.keypoint_2d
                for point in keypoint_2d:
                    # print("    " + str(point))
                    frame = cv2.circle(frame, (int(point[0]),int(point[1])), 5, (0,0,255), 5)
                    body_data.append(point[0])
                    body_data.append(point[1])
                
                key = cv2.waitKey(10)

                if key & 0xFF == ord('l'):
                    #save the data
                    # print(point_data)
                    # print("saving data...")
                    np.savetxt("2DBodyTrainingDataV3.csv", 
                    point_data,
                    delimiter =", ", 
                    fmt ='% s')
                elif key & 0xFF == ord('w'):
                    body_data.append(0)
                elif key & 0xFF == ord('a'):
                    body_data.append(1)
                elif key & 0xFF == ord('s'):
                    body_data.append(2)
                elif key & 0xFF == ord('d'):
                    body_data.append(3)
                elif key & 0xFF == ord('k'):
                    body_data.append(4)
                else:
                    body_data.append(5)



#Timed data collection

                # diff = t0-time.clock()

                # print(f"diff: {diff}")
                # if 0 < diff <= 45: #W
                #     body_data.append(0)
                #     frame = cv2.putText(frame, str('forward'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                #        4, [255,0,0], 2, cv2.LINE_AA)

                # elif 45 < diff <= 90:
                #     body_data.append(1)
                #     frame = cv2.putText(frame, str('left'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                #        4, [255,0,0], 2, cv2.LINE_AA)

                # elif 90 < diff <= 135:    
                #     body_data.append(2)
                #     frame = cv2.putText(frame, str('down'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                #        4, [255,0,0], 2, cv2.LINE_AA)

                # elif 135 < diff <= 180:
                #     body_data.append(3)
                #     frame = cv2.putText(frame, str('right'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                #        4, [255,0,0], 2, cv2.LINE_AA)

                # elif 180 < diff <= 225:
                #     body_data.append(4)
                #     frame = cv2.putText(frame, str('stop'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                #        4, [255,0,0], 2, cv2.LINE_AA)
                # else:
                #     #save the data
                #     print(point_data)
                #     print("saving data...")
                #     np.savetxt("2DBodyTrainingDataV1.csv", 
                #     point_data,
                #     delimiter =", ", 
                #     fmt ='% s')


                point_data.append(body_data)
                point_data.append(body_data)

                # print(body_data)
                # print(np.shape(body_data))

                
                # print("\n Keypoint 3D ")
                keypoint = first_object.keypoint
                
                for point in keypoint:
                    body_data3D.append(point[0])
                    body_data3D.append(point[1])
                    body_data3D.append(point[2])
                    # print("    " + str(point))

                if key & 0xFF == ord('l'):
                    #save the data
                    print(point_data3D)
                    print("saving data...")
                    np.savetxt("3DBodyTrainingDataV3.csv", 
                    point_data3D,
                    delimiter =", ", 
                    fmt ='% s')

                elif key & 0xFF == ord('w'):
                    body_data3D.append(0)
                elif key & 0xFF  == ord('a'):
                    body_data3D.append(1)
                elif key & 0xFF == ord('s'):
                    body_data3D.append(2)
                elif key & 0xFF == ord('d'):
                    body_data3D.append(3)
                elif key & 0xFF == ord('k'):
                    body_data3D.append(4)
                else:
                    body_data3D.append(5)
                point_data3D.append(body_data3D)
                
                # print(body_data3D)
                # print(np.shape(body_data3D))
                cv2.imshow("ZED", frame)
#                 plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# # as opencv loads in BGR format by default, we want to show it in RGB.
#                 plt.show()


    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()