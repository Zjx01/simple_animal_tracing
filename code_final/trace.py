import argparse
from ast import arg
from turtle import color
from threshold_tracing import threshold_calculate
from camshift_tracing import Cam_calculate
from typing import ChainMap
import av
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
import pandas as pd
import PIL.Image as Image


#passing the parameter
def get_parser():
    parser = argparse.ArgumentParser(description="Output tracing plot for the input video")
    #add commands
    parser.add_argument('--input', '-i', default='untitled')
    parser.add_argument('--color','-col', default="black")
    parser.add_argument('--output', '-o', default='trajectory') 
    parser.add_argument('--threshold','-t',default="None")
    parser.add_argument('--thresh1','-t1',default=220)
    parser.add_argument('--thresh2','-t2',default=255)
    parser.add_argument('--column','-c',default=130)
    parser.add_argument('--row','-r',default=90)
    parser.add_argument('--weight','-w',default=80)
    parser.add_argument('--height','-ht',default=80)
    parser.add_argument('--camshift','-cs',default="None")

    return parser     

#%% implementation
if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    frames = args.input #我们所需要读入的文件位置
    threshold_usage = args.threshold
    CamShift_usage =args.camshift
    outputname=args.output
    
    if CamShift_usage == 'True': 
        c=args.column
        h=args.height
        r=args.row
        w=args.weight
        #[int(i) for i in rhcw.split(',')]
        Cam_calculate(frames,r,h,c,w,outputname)

    if threshold_usage == 'True':
        target_color = args.color
        #if the user want specify the threshold themselves
        t1=int(args.thresh1)
        t2=int(args.thresh2)
        color=args.color
        threshold_calculate(frames,color,t1,t2,outputname)
        


