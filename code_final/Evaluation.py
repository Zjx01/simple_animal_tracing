#Evaluatio metrics 

#we compare gt with  the centroids we obtained from camshift and from FIJI 

#read in the fiji tracing result
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def measure_accuracy(ground_truth, test_result,fiji_result):
    ground_truth=pd.read_csv(ground_truth, sep='\t')
    test_result=pd.read_csv(test_result)
    fiji_result=pd.read_csv(fiji_result,sep='\t')

    count_fiji= 0
    count_ours= 0
    for i in range(ground_truth.shape[0]):
        ground_truth_X=np.float64(ground_truth.iloc[i,0].split(" ")[0])
        ground_truth_Y=np.float64(ground_truth.iloc[i,0].split(" ")[1])

        centroids_X_fiji=np.float64(fiji_result.iloc[i,0].split(" ")[0])
        centroids_Y_fiji=np.float64(fiji_result.iloc[i,0].split(" ")[1])
        
        centroids_X_test=test_result.iloc[i,0]
        centroids_Y_test=test_result.iloc[i,1]

        distance_ours=np.sqrt((ground_truth_X-centroids_X_test)**2+(ground_truth_Y-centroids_Y_test)**2)
        distance_fiji=np.sqrt((ground_truth_X-centroids_X_fiji)**2+(ground_truth_Y-centroids_Y_fiji)**2)
        #print(i, centroids_X_fiji,centroids_X_test,centroids_Y_fiji,centroids_Y_test)

        if distance_ours<=6:
            count_ours+=1

        if distance_fiji<=6:
            count_fiji+=1

    accuracy_ours=count_ours/ground_truth.shape[0]
    accuracy_fiji=count_fiji/ground_truth.shape[0]
    return accuracy_ours,accuracy_fiji

if __name__=="__main__":
    ground_truth='/Users/zhaojingxian/Documents/BMI synoptic exam/code/gt_track_all.txt'
    fiji_result='/Users/zhaojingxian/Documents/BMI synoptic exam/code/black_1_short_fiji_track.txt'
    test_result='centroids_mean_shift.csv'
    accuracy= measure_accuracy(ground_truth,test_result,fiji_result)
    print(accuracy)
    #ground_truth=pd.read_csv(ground_truth, sep='\t')


