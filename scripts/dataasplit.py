from utils.datasets import *
import os

rootfolder="/mnt/bigdata/eidur14"

subdir=input("Enter name of folder to split")

while(len(subdir) == 0):
    print("Enter a valid folder name..")
    subdir=input("Enter name of folder to split")

folder=os.path.join(rootfolder,subdir)

print(folder)