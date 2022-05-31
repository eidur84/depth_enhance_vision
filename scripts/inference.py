#!/usr/bin/env python3
# Import ROS libraries and messages

# Import OpenCV libraries and tools

import shutil
from datetime import date, datetime
import cv2
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Reading yolo net
path = '/home/eidur14/trainedweights'

#modelWeights ='depthregression_fromscratch.onnx'
modelWeights ='classdepth_first.onnx'
modelpath=os.path.join(path,modelWeights)

net =  cv2.dnn.readNet(modelpath)
#net = cv2.dnn.readNet((path+modelConfiguration), (path+modelWeights))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Offset classes
classes=['-20','-15','-10','-5','0','5','10','15','20','25','30']

# Global variables
wh = int(640)
confThrsh = 0.45 #0.95
scoreThrsh = 0.4
NMSThrs = 0.4

# Font for puttext funcion
font = cv2.FONT_HERSHEY_PLAIN

## Functions ##

# A function that divides samples into classes based on the offset in
# camera depth from the measured pickpoint
def classnr(camdepthoffset,minval,maxval,binsize):
    binrange=np.arange(minval+binsize,maxval,binsize)
    classval=np.digitize(camdepthoffset,binrange)

    return classval


# Plotting camera offset in data and writing interesting parameters to a table
def plotoffset(offsets,destdir):

    # Variables used in image paths
    suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    imgpath = os.path.join(destdir,"datasetanalysis","camoffset"+suffix+".png")

    # List converted to numpy array and relevant information extracted
    offsets=np.asarray(offsets)
    numsamples=len(offsets)
    minval=np.amin(offsets)
    maxval=np.amax(offsets)
    avgoffset=np.mean(offsets)
    stdoffset=np.std(offsets)

    print(f'# samples: {numsamples} - minimum offset: {minval} - maximum offset {maxval} - average offset {avgoffset} - std offset {stdoffset}')

    # Data for csv prepared
    list_row_append = [(numsamples,minval,maxval,avgoffset,stdoffset)]
 
    dtype = [('Samples in dataset', np.uint),('Min offset', np.float), ('Max offset', np.float),('Average offset', np.float),('Standard deviation', np.float)]
  
    data = np.array(list_row_append, dtype=dtype)

    #data=np.asarray(["D:Fi",avgmeasurederror_depth,avgaccuracy_robot,stdmeasured_depth,avgaccuracy_cam,stdmeasured_cam])

    # Write the results to a csv
    with open(csvpath,'a+') as csvfile:
        np.savetxt(csvfile,data,delimiter=',',fmt=['%d' , '%.2f', '%.2f', '%.2f','%.2f'], comments='')

    plt.figure()
    plt.hist(offsets,np.arange(minval,maxval+1,0.5),color='blue',edgecolor='black',linewidth=1,zorder=3)
    plt.title("MDE offset from pickpoint")
    plt.grid(axis='y',color='black',linestyle='--',linewidth=0.2, zorder=0)
    _ = plt.xlabel('MDE depth offset (mm)')
    _ = plt.ylabel('Number of occurences')
    #plt.xticks(np.arange(minxval, maxxval+1)) 
    #plt.yticks(np.arange(minyval, maxyval+1))
    plt.savefig(imgpath) 
    #plt.show()

# Define a function to show the image in an OpenCV Window
def show_image(img):
    cv2.imshow("Color-image Window", img)
    cv2.waitKey(3)

# Takes in an image from the dataset, gets depth and saves detection image
def getdepth(frame,counter,dest):
    detection=True
    blob = cv2.dnn.blobFromImage(frame, 1.0/255, (wh, wh), [0,0,0], 1, crop=False)
    net.setInput(blob)
    Layernames = net.getLayerNames()
    outputNames = [Layernames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    #outputs = net.forward()
    rows=outputs[0].shape[1]
    #print ("size: ", frame.shape)
    hT, wT, c = frame.shape
    xscale = wT/wh
    yscale = hT/wh

    bbox = []
    classIdx = []
    conf = []
    cls_score=[]

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
                # w, d = int(row[2]*xscale), row[3]/wh
                w, h = int(row[2]*xscale), int(row[3]*yscale)
                print(h)
                x, y = int((row[0]*xscale) - w/2), int((row[1]*yscale) - w/2)
                #print(f'{[x,y,w,d],[classId,float(confidence),class_score]} - confidence: {confidence}')
                #bbox.append([x,y,w,d])
                bbox.append([x,y,w,h])
                classIdx.append(classId)
                conf.append(float(confidence)) 

    idc = cv2.dnn.NMSBoxes(bbox, conf, confThrsh, NMSThrs)
    bbox=np.asarray(bbox)
    conf=np.asarray(conf)
    bbox=bbox[idc]
    conf=conf[idc]
    try:
        winnerbbId = np.argmax(conf)
    except:
        detection=False
        print("No detection")
        
    if detection:
        thebox = bbox[winnerbbId]
        #print(thebox)
        #x,y,w,d = int(thebox[0]), int(thebox[1]), int(thebox[2]), thebox[3]
        x,y,w,h = int(thebox[0]), int(thebox[1]), int(thebox[2]), int(thebox[3])
        # Square bounding box
        xx = int(x+(w/2))# Center point of object
        yy = int(y+(h/2))# Center point of object
        center = (xx,yy)
        
        #coordinates= "x%d y:%d" %(xx,yy)
        #cv2.putText(frame,coordinates,(xx,yy), font, 1,(255,100,255),2,cv2.LINE_4)
        classname=classes[winnerbbId]
        classtext=f'Offset: {classname}'
        #classtext=f'MDE depth: {d:.3f}'
        mdeloc = (30,30)
        #camdepthloc=(30,50)
        #camdepthtext=f'Camera depth: {x:.3f}'
        #cv2.putText(frame,classtext,mdeloc, font, 1,(0,255,255),2,cv2.LINE_4)
        cv2.putText(frame,classtext,mdeloc, font, 1,(0,255,255),2,cv2.LINE_4)
        #cv2.putText(frame,camdepthtext,camdepthloc, font, 1,(0,255,255),2,cv2.LINE_4)
        
        cv2.circle(frame, (xx,yy), 5, (0, 0, 255), 3)
        curr1 = (x,y)
        curr2 = (x+w, y+h)
        cv2.rectangle(frame, curr1, curr2, (0,255,255), 2)
        #show_image(frame)
        cv2.imwrite(dest+"/"+str(counter)+'.png', frame)
        #cv2.waitKey(5000)
        return h
    else:
        return 0

# # This program was written to assemble all gathered data which is stored
# # in seperate folders for each product to one dataset kept in the same folder.
# # Additionally opearations are performed on the lables to fit the needed neural
# # network input format.
# # For Yolo the standard format is: label xcenter ycenter xwidth xheight
# # All coordinates and sizes should be in normalized values from 0-1 based
# # on image resolution, image resolution used for D435 was 848x480 

if __name__ == '__main__':

    # Image width and height for normalization
    imagewidth=848
    imageheight=480

    # Data directory
    rootdir="/mnt/bigdata/eidur14/"

    # Folder to keep the combined dataset in
    destfolder = input("Destination folder name\n")
    while len(destfolder) == 0:
        print("Please enter a valid directory")
        folder = input("Destination folder name\n")

    destdir = os.path.join(rootdir,destfolder)
    destdir_images=os.path.join(destdir,"images")
    destdir_labels=os.path.join(destdir,"labels")

    # Path for writing in csv
    csvpath= os.path.join(destdir,"datasetanalysis","camoffset.csv")

    print(destdir)

    try:
        # Create target Directory
        os.mkdir(destdir)
        print("Directory Created")
    except:
        print("Directory already exists")

    try:
        # Create target Directory
        os.mkdir(os.path.join(destdir,"datasetanalysis"))
        print("Directory Created")
    except:
        print("Directory already exists") 

    srcfolder="newdata"

    try:
        os.mkdir(destdir_images)
    except:
        print("Directory already exists")

    sourcedir = os.path.join(rootdir,"raw",srcfolder)

    print(sourcedir)

    # Counter and lists initialized
    counter=0
    cameraoffset=[]
    classnames=[]
    avgclassoffset=[]

    for subdir, dirs, files in os.walk(sourcedir):
        # Adding subfolder to a vector to spot problematic products
        classnames.append(os.path.basename(subdir))
        depthoffsetvals=[]
        for file in files:
            # Filename at destination folder
            newfn = str(counter).zfill(4)
            # Current filename and extension
            fn, extension = os.path.splitext(file)
            # Source filenames
            sourcetxt=os.path.join(subdir,file)
            sourcecolor=os.path.join(subdir,fn+"color.png")
            sourcedepth=os.path.join(subdir,fn+"depth.png")

            # Loop through the txt files and copy image files in the process.
            # Continue for images
            if extension == ".txt":
                # Read the contents of the label txt file
                with open(sourcetxt) as f:
                    contents = f.readline()
                # Class - xupperleft - yupperleft - width - height - depth - (camera depth) in newdata
                splitter=contents.split()

                # Image read for MDE forward pass
                im2 = cv2.imread(sourcecolor)

                # Image forwarded through the network and returns depth
                mde=getdepth(im2,counter,destdir_images) 

                # Depth from the dataset to compare to estimated depth from NN
                depth=float(splitter[5])
                camdepth=float(splitter[6])
                # If a detection is not made
                if mde > 0:
                    depthoffset=(mde-depth)*1000
                    depthoffsetvals.append(depthoffset)
                    cameraoffset.append(depthoffset)
                    print(f'Camera offset: {depthoffset}')
                else:
                    print("Nothing found")
                    continue
                # Class number based on camera offset: camoffset - min of range - max of range - binsize
                #classval=classnr(depthoffset,-22.5,32.5,5)
                        
            else:
                continue
            
            counter +=1
            print(counter)
        depthoffsetvals=np.asarray(depthoffsetvals)
        avgoffset=np.mean(depthoffsetvals)
        avgclassoffset.append(avgoffset)

    # Removing the parent directory
    avgclassoffset=np.asarray(avgclassoffset[1:])
    classnames=np.asarray(classnames[1:])

    print(avgclassoffset)
    print(classnames)

    # Sorting the array to find the highest and lowest offset
    sortidx=np.argsort(avgclassoffset)
    sortidxflipped=np.flip(sortidx)

    topoverestimate=classnames[sortidx]
    topovervals=avgclassoffset[sortidx]
    topunderestimate=classnames[sortidxflipped]
    topundervals=avgclassoffset[sortidxflipped]

    print(topoverestimate)
    print(topovervals)
    print(topunderestimate)
    print(topundervals)


    # Top and bottom 10 in terms of offsets


    # Offsets between camera depth and measured depth plotted in a histogram
    plotoffset(cameraoffset,destdir)
    
