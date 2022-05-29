#!/usr/bin/env python3
# Import ROS libraries and messages

import rospy
from sensor_msgs.msg import Image, CameraInfo
import pyrealsense2
from geometry_msgs.msg import PoseStamped
# Import OpenCV libraries and tools
import cv2
from cv_bridge import CvBridge, CvBridgeError
#print(cv2.__version__)
import numpy as np                        # fundamental package for scientific computing
import time
import glob
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

path = '/home/eidur14/trainedweights'
print("Environment Ready")
np.set_printoptions(threshold=np.inf)

# Class name    
#classesFile = 'coco.names'
#classNames = []

#with open((path+classesFile), 'rt') as f:
#    classNames = f.read().rstrip('\n').split('\n')
#modelConfiguration ='/home/eidur14/catkin_ws/src/neuralnet/yolov4.cfg' #'yolov4.cfg' # 
#modelConfiguration ='/home/arono16/yolo-obj_new.cfg' #'yolov4.cfg' # 
#modelWeights ='/home/arono16/catkin_ws/src/neuralnet/yolov4.weights'
modelWeights ='depthregression_largestdimension.onnx'

modelpath=os.path.join(path,modelWeights)

#modelConfiguration ='yolov4.cfg' 
#modelWeights ='yolov4.weights' 

#net = cv2.dnn.readNet((modelConfiguration), (modelWeights))
net =  cv2.dnn.readNet(modelpath)
#net = cv2.dnn.readNet((path+modelConfiguration), (path+modelWeights))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Offset classes
classes=['-17.5','-12.5','-7.5','-2.5','2.5','7.5','12.5','17.5','22.5','27.5']

# Global variables
counter = 0
nrAvg = 0
fps = 0.0
timestart = time.time()
timeend = 0.0
elapsed = 0.0
fpsAvg = np.zeros(10)
wh = 640
confThrsh = 0.45 #0.95
scoreThrsh = 0.45
NMSThrs = 0.4

# Font for puttext funcion
font = cv2.FONT_HERSHEY_PLAIN

# Initialize the ROS Node named 'vision', allow multiple nodes to be run with this name
rospy.init_node('vision', anonymous=True)

# Create a publisher that publishes coordinates from camera
pub = rospy.Publisher('pickpoint', PoseStamped, queue_size=5)

# Print "Hello ROS!" to the Terminal and to a ROS Log file located in ~/.ros/log/loghash/*.log
rospy.loginfo("Starting depth subscriber!")

# Initialize the CvBridge class
bridge = CvBridge()
#ERRORY = 75#75 #pxl blue line 
ERRORX = -12

## Functions ##

# Gets fps, uses average over 5 values
def getFps(start, end, nrAvg):
    end = time.time()
    elapsed = end - start
    
    fpsAvg[nrAvg] = int(1 / elapsed)
    nrAvg +=1
    start = time.time()
    if nrAvg == 9:
        nrAvg = 0
    return np.average(fpsAvg), nrAvg, start, end

def findObjects(frame,msg, aligned_info):
    shortestdist = 1
    p = PoseStamped()
    p.header.frame_id = "/camera_color_frame" #"/panda_link8"
    blob = cv2.dnn.blobFromImage(frame, 1.0/255, (wh, wh), [0,0,0], 1, crop=False)
    net.setInput(blob)
    Layernames = net.getLayerNames()
    outputNames = [Layernames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    rows=outputs[0].shape[1]
    #outputs = net.forward()
    print ("size: ", frame.shape)
    hT, wT, c = frame.shape
    xscale = wT/wh
    yscale = hT/wh
    #print(frame.shape)

    bbox = []
    classIdx = []
    conf = []
    cls_score=[]

    bbox = np.array([])
    classIdx = np.array([])
    conf = np.array([])
    cls_score=np.array([])

    #for output in outputs:
    #    for det in output:

    for r in range(rows):
        row=outputs[0][0][r]
        confidence=row[4]
        #print(classId)
        if confidence > confThrsh:
            scores = row[5:] # skip the first 5 values
            classId = np.argmax(scores)
            class_score = scores[classId]
                 
            if class_score > scoreThrsh:
                # w, h = int(det[2]*wT), int(det[3]*hT)
                # x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                #w, h = int(row[2]*xscale), int(row[3]*yscale)
                # Width and depth instead of width and height
                w, d = int(row[2]*xscale), float(row[3]/wh)
                x, y = int((row[0]*xscale) - w/2), int((row[1]*yscale) - w/2)
                #x, y = int((row[0]*xscale) - w/2), int((row[1]*yscale) - h/2)
                bbox.append([x,y,w,d])
                classIdx.append(classId)
                conf.append(float(confidence))
                cls_score.append(class_score)
                # np.append(bbox,[x,y,w,d])
                # np.append(classIdx, classId)
                # np.append(conf,float(confidence))
                # np.append(cls_score,class_score)

    
    idc = cv2.dnn.NMSBoxes(bbox, conf, confThrsh, NMSThrs)
    print(idc.shape)
    print(idc)
    print(conf[idc])
    print(cls_score[idc])
    for i in idc:
        box = bbox[i]
        x,y,w,d = box[0], box[1], box[2], box[3]

        # Square bounding box
        xx = int(x+(w/2))# Center point of object
        yy = int(y+(w/2))# Center point of object
        center = (xx,yy)
        #print("Center: ", center)

        # Show all detected bounding boxes
        cv2.rectangle(frame, (x,y), (x+w,y+w), (0,255,255), 1)

        depth = convert_depth_image(msg, xx, yy)  
        
        # Filter out vacuum bell detections and missing depth points (depth=0)
        if(depth > 0.25 and yy<380):
            
            if(depth < shortestdist):
                shortestdist = depth
                # convert depth to real
                X,Y,Z = convert_depth_to_phys_coord_using_realsense((xx+ERRORX),(yy),depth,aligned_info)  
                currxx = xx
                curryy = yy
                #conftext = "score:%.3f" %(conf[i]*100)
                #send to a topic
                p.header.stamp = rospy.Time.now()
                p.pose.position.x = X
                p.pose.position.y = Y
                p.pose.position.z = Z
                p.pose.orientation.x = x #sending the upper left x coordinate of the box to robot
                p.pose.orientation.y = y #sending the upper left y coordinate of the box to robot
                p.pose.orientation.z = w #sending the width of the box to robot
                p.pose.orientation.w = w #sending the height of the box to robot
                
        
    if p.pose.position.x != 0 and p.pose.position.y != 0 and p.pose.position.z !=0:
        
        #coordinates= "x%d y:%d" %(xx,yy)
        #cv2.putText(frame,coordinates,(xx,yy), font, 1,(255,100,255),2,cv2.LINE_4)
        classname=classes[classIdx[i]]
        #classtext=f'Offset: {classname}'
        classtext=f'MDE depth: {d:.3f}'
        mdeloc = (30,30)
        camdepthloc=(30,50)
        camdepthtext=f'Camera depth: {p.pose.position.x:.3f}'
        cv2.putText(frame,classtext,mdeloc, font, 1,(0,255,255),2,cv2.LINE_4)
        cv2.putText(frame,camdepthtext,camdepthloc, font, 1,(0,255,255),2,cv2.LINE_4)
        
        cv2.circle(frame, (currxx,curryy), 5, (0, 0, 255), 3)
        curr1 = (p.pose.orientation.x,p.pose.orientation.y)
        curr2 = (p.pose.orientation.x+p.pose.orientation.z, p.pose.orientation.y+p.pose.orientation.w)
        cv2.rectangle(frame, curr1, curr2, (0,255,255), 2)
        pub.publish(p)
        rospy.loginfo(p)
    else:
        rospy.logerr("Nothing found")
            

# Gets fps, uses average over 5 values
def getFps(start, end, nrAvg):
    end = time.time()
    elapsed = end - start
    
    fpsAvg[nrAvg] = float(1 / elapsed)
    nrAvg +=1
    start = time.time()
    if nrAvg == 9:
        nrAvg = 0
    return np.average(fpsAvg), nrAvg, start, end

# Define a function to show the image in an OpenCV Window
def show_image(img):
    cv2.imshow("Color-image Window", img)
    cv2.waitKey(3)

def convert_depth_image(ros_image, xcor , ycor):
    bridge = CvBridge()
     # Use cv_bridge() to convert the ROS image to OpenCV format
    try:
     #Convert the depth image using the default passthrough encoding
        depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.float32)
        depth_size = len(depth_array)
        #center_idx = np.array(depth_array.shape) / 2 # center[0] is y value and center[1] is x value
        #print("Convert: ", xcor,",",ycor)
        depth = depth_array[ycor, xcor]/1000.0
        return depth
    except CvBridgeError as e:
        print(e)
     #Convert the depth image to a Numpy array
    return depth

def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
  _intrinsics = pyrealsense2.intrinsics()
  _intrinsics.width = cameraInfo.width
  _intrinsics.height = cameraInfo.height
  _intrinsics.ppx = cameraInfo.K[2]
  _intrinsics.ppy = cameraInfo.K[5]
  _intrinsics.fx = cameraInfo.K[0]
  _intrinsics.fy = cameraInfo.K[4]
  _intrinsics.model = pyrealsense2.distortion.inverse_brown_conrady
  #_intrinsics.model  = pyrealsense2.distortion.none
  _intrinsics.coeffs = [i for i in cameraInfo.D]
  result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
  #result[0]: right, result[1]: down, result[2]: forward
  #return result[0], result[1], result[2]
  return result[2], -result[0], -result[1]

# Define a callback for the Image message
def image_callback(img_msg):
    #print("callback") 
    # log some info about the image topic
    #rospy.loginfo(img_msg.header)
    # Try to convert the ROS Image message to a CV2 Image
    
    rate = rospy.Rate(25) # 10hz#
    global timestart, timeend, nrAvg, counter
    aligned_depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
    aligned_info = rospy.wait_for_message("/camera/aligned_depth_to_color/camera_info", CameraInfo)
    try:
        #print("imgprint")
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    #Find x and y coordinates in pixels
    findObjects(cv_image, aligned_depth_msg, aligned_info) 
    #rospy.loginfo(pos)
    
    # Time and fps calculated
    fps, nrAvg, timestart, timeend = getFps(timestart, timeend, nrAvg)
    hT, wT, c = cv_image.shape
    # Text written, numerical value is seperate from text to keep the text fixed in place
    fpsText = "Fps:%.3f " %(fps)
    #print(fpsText)Q
    cv2.putText(cv_image,fpsText,(10,400), font, 2,(255,255,255),2,cv2.LINE_4)
    framesize = "%.0fx%.0f" %(wT,hT)
    cv2.putText(cv_image,framesize,(10,440), font, 2,(255,0,255),2,cv2.LINE_4)
    show_image(cv_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    # Show the image
    
# Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
sub_image = rospy.Subscriber("/camera/color/image_raw", Image, image_callback, queue_size = 1, buff_size=2**24)
    

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()
