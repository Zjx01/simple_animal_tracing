import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, regionprops_table
import pandas as pd
# 设置初始化的窗口位置
#r,h,c,w = 300,78,40,78 # 设置初试窗口位置和大小 两只小鼠

r,h,c,w = 90,80,130,80 # 设置初试窗口位置和大小 单只小鼠


def Cam_calculate(frames,r,h,c,w,outputname="single_trajectory"):
    track_window = (c,r,w,h)

    cap = cv2.VideoCapture(frames)

    #cap = cv2.VideoCapture('black_1_short.mp4')

    ret, frame= cap.read()

    #shape= frame.shape

    # set the ROI tracking region
    roi = frame[r:r+h, c:c+w]
    #plt.imshow(roi)

    # convert ROI to hsv 
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((180.,66.,36.)))
    mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((180.,255.,45.)))

    # calculate its hsv histogram 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
    roi_hist = cv2.calcHist([hsv_roi],[2],mask,[180],[0,180])

    # Normalization 
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    # set the termination condition if it iterates for 10000 times or move 1 once 设置终止条件，迭代10000次或者至少移动1次
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1 )

    
    column_names=['centroid_x', 'centroid_y', 'mouse_id', 'frame']
    centroids_mean_shift = pd.DataFrame(columns=column_names)
    #mouse id would be collected as  Nan

    while(True):
        
        ret, frame = cap.read()
        if ret == True:
            # calculate the hsv value for each frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # calculate the back project 
            dst = cv2.calcBackProject([hsv],[2],roi_hist,[0,180],1)

            # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            cX = track_window[0]+1/2*track_window[2] 
            cY = track_window[1]+1/2*track_window[3]

            c=pd.DataFrame({'centroid_x':[cX],'centroid_y':[cY],'frame':[ret]})
            centroids_mean_shift = centroids_mean_shift.append(c)
            
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,2)
            cv2.imshow('img2',img2)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        centroids_mean_shift.to_csv('centroids_mean_shift.csv', index=False)
            
    cap.release()
    cv2.destroyAllWindows()

    centroids = pd.read_csv('centroids_mean_shift.csv')


    #plotting
    id_num=len(pd.unique(centroids['mouse_id']))

    #plt.figure(figsize=frame.shape)

    for k in range(id_num):
        id=pd.unique(centroids['mouse_id'])[k]
        #array_xy_id.append(centroids.loc[centroids['mouse_id']==k])
        mouse_x=centroids['centroid_x'].tolist()
        mouse_y=centroids['centroid_y'].tolist()
        plt.scatter(mouse_x,mouse_y,label=k,s=2)
    plt.gca().invert_yaxis()
    plt.savefig(outputname+'.png')
    plt.show()
    

    #save the final result 

#Cam_calculate('black_1_short.mp4',r,h,c,w)
#DFS search for object number 
# def numIslands(grid: [[str]]) -> int:
#         def dfs(grid, i, j):
#             if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] == 0: return
#             grid[i][j] = 0
#             dfs(grid, i + 1, j)
#             dfs(grid, i, j + 1)
#             dfs(grid, i - 1, j)
#             dfs(grid, i, j - 1)
#             dfs(grid, i + 1, j+1)
#             dfs(grid, i-1, j + 1)
#             dfs(grid, i - 1, j-1)
#             dfs(grid, i-1, j - 1)
#         count = 0
#         for i in range(len(grid)):
#             for j in range(len(grid[0])):
#                 if grid[i][j] == 255:
#                     dfs(grid, i, j)
#                     count += 1
#         return count

# numIslands(mask)


#this is not implementable as some pixels of the mouse are isolated from body 

