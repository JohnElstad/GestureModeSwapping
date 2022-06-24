# Generates heatmap data for a single gesture and saves the data to a csv file. 
# The CSV file contains x and z data using zed object detections as well as the gesture predicted by custom model
# to use, walk around using ONLY the gesture you want to test to all points you want data for in the x,z plane infront of zed camera.
# Once you are satisfied with the amount of data collected, have someone press L. This saves the data to the location set by 'savePath' seen below in main(). This can then be
# imported into heatmap.py to create the heatmap. Make sure you only press L once to save the data and that the opencv window is still recording
# when you press L. If it isn't, the data won't be saved. 
# John Elstad 6/6/2022

from traceback import FrameSummary
from tracemalloc import Frame
import pyzed.sl as sl
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt

def main():
    savePath = "Heatmap3DLeftV1Out.csv" #where your data gets saved to

    # Create a Camera object
    zed = sl.Camera()

    runtime = sl.RuntimeParameters()
    #Load Custom Model
    model = load_model('/home/user/john_ws/src/gesture_ctrl_john/src/2DBodyModelV4XGesture.h5')
    model.summary()

    #mat added 
    mat = sl.Mat()
    
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
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
    body_data3DX = []
    body_data3DZ = []
    body_data3DPrediction = []
    


    t0 = time.clock()
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        err = zed.retrieve_objects(objects, obj_runtime_param)
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
                person = first_object
                body_data = []
                

                
                # print(" Keypoint 2D ")
                keypoint_2d = first_object.keypoint_2d
                for point in keypoint_2d:
                    # print("    " + str(point))
                    frame = cv2.circle(frame, (int(point[0]),int(point[1])), 5, (0,0,255), 5)
                    body_data.append(point[0])
                    body_data.append(point[1])

#Run Model And Predict from input Data
                results = model.predict(np.array(body_data).reshape(1,-1))
                confidence = np.max(results)
                prediction = np.argmax(results)
                frame = cv2.putText(frame, str(prediction), (int(person.keypoint_2d[0][0]),int(person.keypoint_2d[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 5, cv2.LINE_AA)

#3D model

                point_data.append(body_data)
                point_data.append(body_data)

                # print(body_data)
                # print(np.shape(body_data))
                
                point = first_object.position

                
                # print("    " + str(point))

                if key & 0xFF == ord('l'):
                    #save the data
                    point_data3D.append(body_data3DX)
                    point_data3D.append(body_data3DZ)
                    point_data3D.append(body_data3DPrediction)
                    print(point_data3D)
                    print("saving data...")
                    np.savetxt(savePath, 
                    point_data3D,
                    delimiter =", ", 
                    fmt ='% s')

                else:
                    body_data3DX.append(point[0])
                    body_data3DZ.append(point[2])
                    body_data3DPrediction.append(prediction)
                    print(body_data3DX)
                    print(body_data3DZ)
                    print(body_data3DPrediction)
                
                # print(body_data3D)
                # print(np.shape(body_data3D))
                cv2.imshow("ZED", frame)



    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()