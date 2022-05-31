#!/usr/bin/env python3
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime

# This program was written to assemble all gathered data which is stored
# in seperate folders for each product to one dataset kept in the same folder.
# Additionally opearations are performed on the lables to fit the needed neural
# network input format.
# For Yolo the standard format is: label xcenter ycenter xwidth xheight
# All coordinates and sizes should be in normalized values from 0-1 based
# on image resolution, image resolution used for D435 was 848x480 

# Path to save plots and data for tables
folder_path = "/home/lab/Documents/Eidur/Myndir/"

csvpath= os.path.join(folder_path,"datasetanalysis","camoffset.csv")


# A function that divides samples into classes based on the offset in
# camera depth from the measured pickpoint
def classnr(camdepthoffset,minval,maxval,binsize):
    binrange=np.arange(minval+binsize,maxval,binsize)
    classval=np.digitize(camdepthoffset,binrange)

    return classval


# Plotting camera offset in data and writing interesting parameters to a table
def plotoffset(offsets):

    # Variables used in image paths
    suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    imgpath = os.path.join(folder_path,"datasetanalysis","camoffset"+suffix+".png")

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
    plt.title("Camera depth offset from pickpoint")
    plt.grid(axis='y',color='black',linestyle='--',linewidth=0.2, zorder=0)
    _ = plt.xlabel('Camera depth offset (mm)')
    _ = plt.ylabel('Number of occurences')
    #plt.xticks(np.arange(minxval, maxxval+1)) 
    #plt.yticks(np.arange(minyval, maxyval+1))
    plt.savefig(imgpath) 
    plt.show()

if __name__ == '__main__':

    # Image width and height for normalization
    imagewidth=848
    imageheight=480

    # Data directory
    rootdir="/home/lab/Documents/Eidur/Data/"

    # Folder to keep the combined dataset in
    destfolder = input("Destination folder name\n")
    while len(destfolder) == 0:
        print("Please enter a valid directory")
        folder = input("Destination folder name\n")

    destdir = os.path.join(rootdir,destfolder)
    destdir_images=os.path.join(destdir,"images")
    destdir_labels=os.path.join(destdir,"labels")

    print(destdir)

    try:
        # Create target Directory
        os.mkdir(destdir)
        print("Directory Created ")
    except:
        print("Directory already exists") 

    # Dataset source folder
    # folder = input("Source folder name\n")
    # while len(folder) == 0:
    #     print("Please enter a valid directory")
    #     folder = input("Destination folder name\n")
    srcfolder="newdata"

    try:
        os.mkdir(destdir_images)
    except:
        print("Directory already exists")

    try:
        os.mkdir(destdir_labels)
    except:
        print("Directory already exists")

    sourcedir = os.path.join(rootdir,srcfolder)

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

                # Variables read from the text file

                # Fixing coordinates to middle of object and normalizing to values between 0-1
                newx=(float(splitter[1])+float(splitter[3])/2)/imagewidth
                newy=(float(splitter[2])+float(splitter[4])/2)/imageheight

                # Approaches to how best use the width parameter of the network to represent
                # The dimensions of the object since the height parameter is occupied by depth

                #Encoding only the larger dimension of the box to replace height with depth.
                # Divided by horizontal dimension.
                # if float(splitter[3]) >= float(splitter[4]):
                #     newwidth=float(splitter[3])/imagewidth
                # else:
                #     newwidth=float(splitter[4])/imageheight

                # newwidth=float(splitter[3])/imagewidth

                # No bounding box necessary, only use mid point for pick point

                # Radius around the rectangle divided by horiontal resolution used for width parameter
                #newwidth=(np.sqrt(float(splitter[3])**2+float(splitter[4])**2)/2)/imagewidth

                #print(f'w:{float(splitter[3])} - h: {float(splitter[4])} - radius={newwidth}')

                newwidth=float(splitter[3])/imagewidth
                newheight=float(splitter[4])/imageheight
                depth=float(splitter[5])
                camdepth=float(splitter[6])
                # Offset of camera depth from pick point
                depthoffset=(camdepth-depth)*1000
                depthoffsetvals.append(depthoffset)
                cameraoffset.append(depthoffset)
                # Class number based on camera offset: camoffset - min of range - max of range - binsize
                classval=classnr(depthoffset,-22.5,32.5,5)

                #print(f'Camera offset: {depthoffset} - Class assigned: {classval}')

                # Writing fixed labels to destination
                with open(os.path.join(destdir_labels,newfn+".txt"), 'w') as f:
                    # Estimating offset using class
                    f.write(f'{classval} {newx:.6f} {newy:.6f} {newwidth:.6f} {newheight:.6f}')
                    # Encoding depth using YOLO height parameter - single class
                    #f.write(f'0 {newx:.6f} {newy:.6f} {newwidth:.6f} {depth:.6f}')
                    #f.write(f'0 {newx:.6f} {newy:.6f} {newwidth:.6f} {newheight:.6f}')
                    #print(f'{classval} {newx:.6f} {newy:.6f} {newwidth:.6f} {newheight:.6f} {depth:.6f}')

                try:
                    # Files copied to destination directory
                    #shutil.copy(sourcetxt, os.path.join(destdir,newfn+".txt"))
                    shutil.copy(sourcecolor, os.path.join(destdir_images,newfn+".png"))
                    #shutil.copy(sourcedepth, os.path.join(destdir,newfn+"depth.png"))

                    counter+=1
                
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                
                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                
                # For other errors
                except:
                    print("Error occurred while copying file.") 
                        
            else:
                continue
            
            print(counter)
    #     depthoffsetvals=np.asarray(depthoffsetvals)
    #     avgoffset=np.mean(depthoffsetvals)
    #     avgclassoffset.append(avgoffset)

    # # Removing the parent directory
    # avgclassoffset=np.asarray(avgclassoffset[1:])
    # classnames=np.asarray(classnames[1:])

    # print(avgclassoffset)
    # print(classnames)

    # # Sorting the array to find the highest and lowest offset
    # sortidx=np.argsort(avgclassoffset)
    # sortidxflipped=np.flip(sortidx)

    # print(sortidx)

    # topoverestimate=classnames[sortidx]
    # topunderestimate=classnames[sortidxflipped]

    # print(topoverestimate)
    # print(topunderestimate)

    # # Top and bottom 10 in terms of offsets


    # # Offsets between camera depth and measured depth plotted in a histogram
    # plotoffset(cameraoffset)