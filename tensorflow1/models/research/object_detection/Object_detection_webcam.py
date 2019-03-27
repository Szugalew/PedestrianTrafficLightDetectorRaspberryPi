

import numpy
numpy.version.version


# Import packages
import os
import sys
#print (sys.path)


import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf

import sys

import time
import board
import digitalio


ledL = digitalio.DigitalInOut(board.D13)
ledL.direction = digitalio.Direction.OUTPUT
 
ledM = digitalio.DigitalInOut(board.D12)
ledM.direction = digitalio.Direction.OUTPUT

ledR = digitalio.DigitalInOut(board.D18)
ledR.direction = digitalio.Direction.OUTPUT


IM_WIDTH = 1280
IM_HEIGHT = 720
# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 2

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Define left box coordinates (top left and bottom right)
TL_left = (int(IM_WIDTH*0.01),int(IM_HEIGHT*0.01))
BR_left = (int(IM_WIDTH*0.33),int(IM_HEIGHT*0.99))

# Define middle box coordinates (top left and bottom right)
TL_middle = (int(IM_WIDTH*0.34),int(IM_HEIGHT*0.01))
BR_middle = (int(IM_WIDTH*0.66),int(IM_HEIGHT*0.99))

# Define inside box coordinates (top left and bottom right)
TL_right = (int(IM_WIDTH*0.67),int(IM_HEIGHT*0.01))
BR_right = (int(IM_WIDTH*0.99),int(IM_HEIGHT*0.99))


if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    

for i in range (2):
    ledR.value = True
    time.sleep(0.5)
    ledR.value = False
    
    ledM.value = True
    time.sleep(0.5)
    ledM.value = False
    
    ledL.value = True
    time.sleep(0.5)
    ledL.value = False
    time.sleep(0.5)



# Continuously capture frames and perform object detection on them
for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    t1 = cv2.getTickCount() 
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = frame1.array
    frame.setflags(write=1)
    
#added bgr to rgb	
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})


    
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.10)
#############
    cv2.rectangle(frame,TL_left,BR_left,(255,20,20),3)
    cv2.putText(frame,"Left box",(TL_left[0]+10,TL_left[1]-10),font,1,(255,20,255),3,cv2.LINE_AA)
    cv2.rectangle(frame,TL_middle,BR_middle,(20,20,255),3)
    cv2.putText(frame,"Middle box",(TL_middle[0]+10,TL_middle[1]-10),font,1,(20,255,255),3,cv2.LINE_AA)
    cv2.rectangle(frame,TL_right,BR_right,(255,20,20),3)
    cv2.putText(frame,"Right box",(TL_right[0]+10,TL_right[1]-10),font,1,(255,20,255),3,cv2.LINE_AA)






    
##############
	# Draw FPS
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
	
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
	
	# FPS calculation
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
		
		
    
    objects = []
    for index, value in enumerate(classes[0]):
      object_dict = {}
      if scores[0, index] > 0.25:
        object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                            scores[0, index]
        objects.append(object_dict)

    things = str(objects).strip('[]')
    if objects:
        things = str(objects[0])
    if 'walk' not in things :
        print("DO NOT WALK.....", things)
        ledR.value = True
        ledL.value = True
        ledM.value = False
        time.sleep(0.1)
        ledR.value = False
        ledM.value = False
        ledL.value = False
    else:
        print("Yeah you can walk", things)
        
        if ( ( int(classes[0][0]) == 1) or ( int(classes[0][0] == 2) ) ):
            x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
            y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)

        # Draw a circle at center of object
            cv2.circle(frame,(x,y), 5, (75,13,180), -1)

        # If object is in left box
            if ((x > TL_left[0]) and (x < BR_left[0]) and (y > TL_left[1]) and (y < BR_left[1])):
                ledR.value = False
                ledL.value = True
                ledM.value = False

        # If object is in middle box
            if ((x > TL_middle[0]) and (x < BR_middle[0]) and (y > TL_middle[1]) and (y < BR_middle[1])):
                ledR.value = False
                ledL.value = False
                ledM.value = True
        
        # If object is in right box
            if ((x > TL_right[0]) and (x < BR_right[0]) and (y > TL_right[1]) and (y < BR_right[1])):
                ledR.value = True
                ledL.value = False
                ledM.value = False

        
        
        
########################################################

   
    
    print()







    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
	
    rawCapture.truncate(0)
camera.close()
# Clean up
cv2.destroyAllWindows()
ledR.value = False
ledL.value = False
ledM.value = False
