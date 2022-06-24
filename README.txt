This is the code for my capstone project to train a neural network that can recognize different gestures. The project uses ZED API to recognize body pose and predict gestures based on the pose. More information can be found in the capstone report. EVERYTHING IN THE FOLDER IS UPDATED. Ignore everything outside of the folder. They are remnents from the beginning of this project.

6/23/2022 - John Elstad
Contains code that can swap between two predefined modes ("Follow Me Mode" and "Gesture Control Mode") using 6 predefined gesetures. The code recognizes the gestures using a custom tensorflow model and ZED api. Publishes move commands to ROS depending on the current mode. Mode can be adjusted with the Mode swap gesture. 

Gestures trained are:
- No Gesture
- Right
- Left
- Back
- Forward
- Mode Swap/Arm Cross

Details about how to use the gestures are in the report.

The code also includes two heatmap programs for mapping the effective range of the gesture algorithm. One program is for collecting data and the other is for actually generating the heatmap.

The .h files are the saved custom models. These were trained by me and get imported into the mode swapping code. They recognize the gesture from the body pose estimations. 




