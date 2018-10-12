# This code is used to test the relationship of diffrent Histogram Similarity with sAUC (test on SALICON Validation Images)
import cv2
import tensorflow as tf 
import numpy as np 
from matplotlib import pyplot as plt 
import os 
import scipy.io as sio  # for saving .MAT file
# from matplotlib import cm as cm
'''
   similarity calculation methods: (int)
   0: CV_COMP_CORREL Correlation
   1: CV_COMP_CHISQR Chi-Square
   2: CV_COMP_CHISQR_ALT Alternative Chi-Square
    CV_COMP_INTERSECT Intersection
    CV_COMP_BHATTACHARYYA Bhattacharyya distance
    CV_COMP_HELLINGER Synonym for CV_COMP_BHATTACHARYYA
    CV_COMP_KL_DIV Kullback-Leibler divergence
'''



img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_2/'
# txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_name_list.txt'
txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_name_list_sorted.txt'

index = 0
file = open(txt_path, 'r')

score_0_vec = []
score_1_vec = []
score_2_vec = []
score_3_vec = []
score_4_vec = []


for i in range(0, 500, 1): # 500
    index += 1
    name = file.readline()
    # print("index is :", index)
    print("name is :", name)
    img_GT_path = img_path + name + '_GT.jpg'
    img_SM_path = img_path + name + '_SM.jpg'
    
    img_GT = cv2.imread(img_GT_path, cv2.IMREAD_GRAYSCALE)
    img_SM = cv2.imread(img_SM_path, cv2.IMREAD_GRAYSCALE)
    

    bins = np.arange(256).reshape(256,1)
    # print("bins is :", bins)
    hist_item1 = cv2.calcHist([img_GT],[0],None,[256],[0,256])
    hist_item2 = cv2.calcHist([img_SM],[0],None,[256],[0,256])
    # print("hist1 is :", hist_item1)
    cv2.normalize(hist_item1,hist_item1,0,255,cv2.NORM_MINMAX)
    cv2.normalize(hist_item2,hist_item2,0,255,cv2.NORM_MINMAX)
    # print("normalized hist1 is :", hist_item1)
    # print("normalized hist2 is :", hist_item2)
    score_0 = cv2.compareHist(hist_item1, hist_item2, 0)
    print("score0 is :", score_0)
    score_0_vec.append(score_0)

    score_1 = cv2.compareHist(hist_item1, hist_item2, 1)
    print("score1 is :", score_1)
    score_1_vec.append(score_1)
    

    score_2 = cv2.compareHist(hist_item1, hist_item2, 2)
    print("score2 is :", score_2)
    score_2_vec.append(score_2)

    score_3 = cv2.compareHist(hist_item1, hist_item2, 4)
    print("score3 is :", score_3)
    score_3_vec.append(score_3)
    
    '''
    score_4 = cv2.compareHist(hist_item1, hist_item2, 4)
    print("score4 is :", score_4)
    score_4_vec.append(score_4)
    '''
file.close()
# print("vec score 0 is:", score_0_vec)


###### The following part is for drawing figures (The test performance is not satisfing)

score_1_vec = np.array(score_1_vec).reshape(1,500)
cv2.normalize(score_1_vec,score_1_vec,0,1,cv2.NORM_MINMAX)
# score_1_vec = 1 - score_1_vec
score_2_vec = np.array(score_2_vec).reshape(1,500)
cv2.normalize(score_2_vec,score_2_vec,0,1,cv2.NORM_MINMAX)
score_2_vec = 1 - score_2_vec

score_3_vec = np.array(score_3_vec).reshape(1,500)
cv2.normalize(score_3_vec,score_3_vec,0,1,cv2.NORM_MINMAX)
score_3_vec = 1 - score_3_vec


# cv2.normalize(score_2_vec,score_2_vec,0,1,cv2.NORM_MINMAX)
# cv2.normalize(score_3_vec,score_3_vec,0,1,cv2.NORM_MINMAX)
# print("score_1_vec is :", score_1_vec)

load_data = sio.loadmat('/media/ubuntu/Ubuntu 16.0/SaliencyResult/1to500imgsScores/sAUC.mat')
load_matrix = load_data['AUC_vec_s']
# print(load_matrix)

# LEN = 25
LEN = 40
X = range(0,LEN)

Y = []
for i in range(0, LEN, 1):
    Y.append(load_matrix[0, i])
# print(Y)

Y1 = []
score_0_vec = np.array(score_0_vec).reshape(1,500)
for i in range(0, LEN, 1):
    Y1.append(score_0_vec[0, i])

Y2 = []
score_1_vec = np.array(score_1_vec).reshape(1,500)
for i in range(0, LEN, 1):
    Y2.append(score_1_vec[0, i])

Y3 = []
score_2_vec = np.array(score_2_vec).reshape(1,500)
for i in range(0, LEN, 1):
    Y3.append(score_2_vec[0, i])

Y4 = []
score_3_vec = np.array(score_3_vec).reshape(1,500)
for i in range(0, LEN, 1):
    Y4.append(score_3_vec[0, i])



plt.figure(figsize=(14,8)) 
plt.title('Relationship between sAUC and Histogram Similarity')
# plt.plot(X, Y, label='Frist line',linewidth=1,color='b',marker='+', markerfacecolor='red',markersize=1) 
plt.plot(X, Y, linewidth=3,color='b',marker='o',markersize=10, label='sAUC') 
plt.plot(X, Y1, linewidth=3,color='r',marker='o',markersize=10, label='Histogram Correlation') 
plt.plot(X, Y2, linewidth=3,color='g',marker='o',markersize=10, label='Chi-Square Distance') 
plt.plot(X, Y3, linewidth=3,color='y',marker='o',markersize=10, label='Alternative Chi-Square Distance') 
# plt.plot(X, Y4, linewidth=1,color='cyan',marker='.',markersize=1, label='Bhattacharyya distance') 
# plt.plot(X, Y3, label='Frist line',linewidth=1,color='g',marker='+', markerfacecolor='blue',markersize=1) 

# plt.plot(X,Y2,"b--",linewidth=1)
# plt.plot(X,Y)
# plt.plot(X,Y,"r--",linewidth=2)

# plt.plot(X, Y, label='Frist line',linewidth=1, color='r',marker='.', markerfacecolor='blue',markersize=24) 
# plt.plot(X, Y2, label='second line',linewidth=1, color='b',marker='.', markerfacecolor='red',markersize=24)
# plt.plot(X,Y,label='second line') 
plt.legend()
plt.xlabel('Image Index')
plt.ylabel('Performance Score')
plt.show()


Big_Matrix = np.zeros([LEN, 5])
# print(Big_Matrix)
Big_Matrix[:,0] = Y
Big_Matrix[:,1] = Y1
Big_Matrix[:,2] = Y2
Big_Matrix[:,3] = Y3
Big_Matrix[:,4] = Y4
# print(Big_Matrix)
Big_Matrix_2 = Big_Matrix[Big_Matrix[:,0].argsort()] # sort Big_Matrix corresponding to the first column
# print("Sorted Big_Matrix_2 is :", Big_Matrix_2)

Col_0 = Big_Matrix_2[:,0]
Col_1 = Big_Matrix_2[:,1]
Col_2 = Big_Matrix_2[:,2]
Col_3 = Big_Matrix_2[:,3]
Col_4 = Big_Matrix_2[:,4]
# print(Col_1)

plt.figure(figsize=(14,8)) 
# plt.plot(X, Y, label='Frist line',linewidth=1,color='b',marker='+', markerfacecolor='red',markersize=1) 
plt.plot(X, Col_0, linewidth=3,color='b',marker='o',markersize=10, label='sAUC Score') 
plt.plot(X, Col_1, linewidth=3,color='r',marker='o',markersize=10, label='Histogram Correlation') 
plt.plot(X, Col_2, linewidth=3,color='g',marker='o',markersize=10, label='Chi-Square Distance') 
plt.plot(X, Col_3, linewidth=3,color='y',marker='o',markersize=10, label='Alternative Chi-Square Distance') 
# plt.plot(X, Col_4, linewidth=1,color='cyan',marker='.',markersize=1) 

plt.legend()
plt.xlabel('Image Index')
plt.ylabel('Performance Score (Sorted By sAUC)')
plt.show()
