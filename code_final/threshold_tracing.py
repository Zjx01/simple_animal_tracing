from typing import ChainMap
import av
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
import pandas as pd
import PIL.Image as Image



'''
Dectection on the first image

# Open the video, contain audio and vedio
container = av.open("Mice_short.mp4")
# Extract video stream
video = container.streams.video[0]
# Decode video frames
frames = container.decode(video)

#see the first frame of one test image 
#frame0=next(frames)
#frame0 = np.array(frame0.to_image())
# print(np.max(frame0))
# plt.imshow(frame0)
# plt.show()

#convert it to gray scale
# frame0_gray=cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
#black to white,white to black

# invert_frame0=1.0-frame0_gray
# fix,ax=plt.subplots(1,3,figsize=(15,5))
# ax[0].imshow(frame0)
# ax[1].imshow(frame0_gray,cmap="gray")
# ax[2].imshow(invert_frame0,cmap="gray")
'''


def threshold_calculate(frames,recalculate=True,target_color="black",t1=220,t2=255,outputname="trajectory"):

    #Open the video, contain audio and vedio
    container = av.open(frames)
    # Extract video stream
    video = container.streams.video[0]
    # Decode video frames
    frames = container.decode(video)

    # # The column names for our DataFrame
    column_names=['centroid_x', 'centroid_y', 'mouse_id', 'frame']
    centroids = pd.DataFrame(columns=column_names)

    if recalculate:
        kernel=np.ones((5,5),np.uint8) #kernel for dilation and erosion
        # Iterate through frames
        for m, f in enumerate(frames):
            # Convert to Numpy
            f = np.array(f.to_image())
            # Convert to grayscale
            im_gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)

            #for autothresholding based on mean-shift algorithm
            denoised_im_gray=median(im_gray)
            
            #erosion and dilation process:
            erosion=cv2.erode(im_gray,kernel,iterations=5)
            dilation=cv2.dilate(erosion,kernel,iterations=5)
            # cv2.imshow("pf",dilation)

            #color of target 
            if target_color=="black":
                target_dilation=255-dilation 

            else:
                target_dilation=dilation
            
            #mannual threhsholding tracing 
            ret, thresh = cv2.threshold(target_dilation, t1, t2, cv2.THRESH_BINARY) # 阈值处理 二值化 
            thresh1 = cv2.GaussianBlur(thresh,(3,3),0)# 高斯滤波
            #contours,hirearchy=cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# 找出连通域
            contours,hirearchy=cv2.findContours(thresh1,mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            area=[] #建立空数组，放连通域面积
            contours1=[]   #建立空数组，放减去后的数组
            for i in contours:
                area.append(cv2.contourArea(i))
                # print(area)
                if cv2.contourArea(i)>100:  # 计算面积 去除面积小的 连通域
                    contours1.append(i)
            #print(len(contours1)-1) #计算连通域个数
            draw=cv2.drawContours(f,contours1,-1,(0,255,0),1) #描绘连通域

            
            for k,j in zip(contours1,range(len(contours1))):
                M = cv2.moments(k)
                cX=int(M["m10"]/M["m00"])
                cY=int(M["m01"]/M["m00"])
                draw1=cv2.putText(draw, str(j), (cX, cY), 1,1, (255, 0, 255), 1) #在中心坐标点上描绘数字
                #cv2.imshow("draw",draw1)
                #cv2.imshow("thresh1",thresh1)
                #cv2.waitKey()
                #cv2.destroyWindow()
                
                #save the location information in the pd table
                c=pd.DataFrame({'centroid_x':[cX],'centroid_y':[cY],'mouse_id':[str(j)]})
                c['frame'] = [m]
            # Append to our DataFrame
                centroids = centroids.append(c)
        # Save centroids to CSV
        centroids.to_csv('centroids.csv', index=False)
  
    centroids = pd.read_csv('centroids.csv')

    #plotting
    colors = centroids['mouse_id'].values
    id_num=len(pd.unique(centroids['mouse_id']))


    #plt.figure(figsize=next(frames).shape)
    for k in range(id_num):
        id=pd.unique(centroids['mouse_id'])[k]
        mouse_trace_data=centroids.loc[centroids["mouse_id"]==id]
        mouse_x=mouse_trace_data['centroid_x'].tolist()
        mouse_y=mouse_trace_data['centroid_y'].tolist()
        #plt.scatter(mouse_x,mouse_y,label=k,s=2)        
        plt.scatter(mouse_x,mouse_y,label=k,s=2)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(outputname+'.png')
    plt.show()
    #plt.savefig()


#threshold_calculate('Mice_short.mp4')