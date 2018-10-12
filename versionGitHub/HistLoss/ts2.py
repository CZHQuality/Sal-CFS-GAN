import matplotlib.pyplot as plt 
import scipy.io as sio
'''
y1=[10,13,5,40,30,60,70,12,55,25] 
x1=range(0,10) 
x2=range(0,10) 
y2=[5,8,0,30,20,40,50,10,40,15] 
plt.plot(x1,y1,label='Frist line',linewidth=3,color='r',marker='o', 
         markerfacecolor='blue',markersize=12) 
plt.plot(x2,y2,label='second line') 
plt.xlabel('Plot Number') 
plt.ylabel('Important var') 
plt.title('Interesting Graph\nCheck it out') 
plt.legend() 
plt.show() 
'''

x1 = range(0,500)
load_data = sio.loadmat('/media/ubuntu/Ubuntu 16.0/SaliencyResult/1to500imgsScores/sAUC.mat')
load_matrix = load_data['AUC_vec_s']
print("load_matrix is:", load_matrix)
# y1 = load_matrix
# y1 = y1.tolist()

y1 = []
for i in range(0, 500, 1):
    y1.append(load_matrix[0, i])

print(y1)
plt.plot(x1,y1,label='Frist line',linewidth=3,color='r',marker='o', markerfacecolor='blue',markersize=12) 
plt.legend() 
plt.show() 