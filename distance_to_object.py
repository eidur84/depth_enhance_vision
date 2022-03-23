import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
#import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import time
import glob
import math
print("Environment Ready")
np.set_printoptions(threshold=np.inf)

# Font for puttext funcion
font = cv2.FONT_HERSHEY_PLAIN

# Global variables

nrAvg = 0
fps = 0.0
timestart = time.time()
timeend = 0.0
elapsed = 0.0
fpsAvg = np.zeros(10)
wh = 416
confThrsh = 0.5
NMSThrs = 0.3

# Gets fps, uses average over 5 values
def getFps(start, end, nrAvg, display: bool):
    end = time.time()
    elapsed = end - start
    if display:
        print(elapsed)
    fpsAvg[nrAvg] = int(1 / elapsed)
    nrAvg +=1
    start = time.time()
    if nrAvg == 9:
        nrAvg = 0
    return np.average(fpsAvg), nrAvg, start, end

def findObjects(frame, depthframe):
    #print("TEST")
    #print(frame.shape, type(frame), frame.dtype)
    # color_image converted to blob for dnn processing and set as net input
    blob = cv2.dnn.blobFromImage(frame, 1/255, (wh, wh), [0,0,0], 1, crop=False)
    #print(blob.shape, type(blob), blob.dtype)
    #print("blob", blob)
    net.setInput(blob)
    Layernames = net.getLayerNames()
    #print("Layernames ", Layernames)
    outputNames = [Layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print("Outnames", outputNames)
    outputs = net.forward(outputNames)
    
    
    hT, wT, c = frame.shape
    #print("ht", hT)
    #print("wt", wT)
    expected = 300
    aspect = wT / hT
    resized_image = cv2.resize(frame, (round(expected * aspect), expected))
    crop_start = round(expected * (aspect - 1) / 2)
    crop_img = resized_image[0:expected, crop_start:crop_start+expected]

    bbox = []
    classIdx = []
    conf = []
    #print("output",outputs[0][39])
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThrsh:
                #print("output", output)
                #print("det", scores)
                #print("classID:", classId)
                #print("conf", confidence)
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIdx.append(classId)
                conf.append(float(confidence)) 
   
    idc = cv2.dnn.NMSBoxes(bbox, conf, confThrsh, NMSThrs)
    f = 480/(math.tan(math.radians(35)))
    for i in idc:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]

        xx = int(x+(w/2))# Center point of object
        yy = int(y+(h/2))# Center point of object
        print("X: %f, Y: %f. " % (xx,yy))
        #Get distance to item from camera 
        
        
        dist = depthframe.get_distance(xx,yy)
        xreal = dist * ((xx-(wT/2))/f)
        yreal = dist * ((yy-(hT/2))/f)
        #print("X: %f, Y: %f. " % (xreal,yreal))
        distanceText = "Distance: %.3fm (x:%.3fm, y:%.3fm)" %(dist, xreal, yreal)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        cv2.putText(frame, f'{classNames[classIdx[i]].upper()} {conf[i]*100}%', \
                (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 255), 2)
        cv2.putText(frame, distanceText , (xx,yy), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0,255, 255), 2)
        center = (xx,yy)
        # Puts in a circle where the center of the object is 
        cv2.circle(frame, center, 5, (255, 0, 0), 2)

# Class names

classesFile = 'coco.names'
#classesFile = 'obj_new.names'

classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Setup:
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config= rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # 960 ig 540
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
print("check the version opencv.")
print(cv2.__version__)
print(cv2.__file__)
try:
    
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        #if not depth_frame or not color_frame:
            #   continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        
        
        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        
        # Update color and depth frames:
        aligned_depth_frame = frames.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())       

        
        # Time and fps calculated
        fps, nrAvg, timestart, timeend = getFps(timestart, timeend, nrAvg, False)

        # Objects found
        findObjects(color_image, aligned_depth_frame)

        # Text written, numerical value is seperate from text to keep the text fixed in place
        fpsText = "Fps:%.1f " %(fps)
        cv2.putText(color_image,fpsText,(10,675), font, 2,(255,255,255),2,cv2.LINE_4)
        framesize = "%.0fx%.0f" %((color_image.shape)[0],(color_image.shape)[1])
        cv2.putText(color_image,framesize,(10,715), font, 2,(255,255,255),2,cv2.LINE_4)
        alignedTogether = np.hstack((colorized_depth, color_image))
        #Show images
        #cv2.namedWindow('alignedTogether', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('alignedTogether', alignedTogether)
    
        # Display the resulting color_image
        cv2.imshow('Rammi', color_image)

        #Show images
        #cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('Image', images)

        #Show images
        #cv2.namedWindow('Depth COLORIZED', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('Depth COLORIZED', colorized_depth)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



finally:

    # Stop streaming
    pipeline.stop()
