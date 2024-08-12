import numpy as np
import matplotlib.pyplot as plt

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

A1=np.loadtxt('1.csv',delimiter=',')
A2=np.loadtxt('2.csv',delimiter=',')
A3=np.loadtxt('3.csv',delimiter=',')
A4=np.loadtxt('4.csv',delimiter=',')
A5=np.loadtxt('5.csv',delimiter=',')
A6=np.loadtxt('6.csv',delimiter=',')

A1=np.delete(A1,A1[:,0]>0.8,0)

A2=np.delete(A2,A2[:,0]<0.8,0)
A2=np.delete(A2,A2[:,0]>1.6,0)

A3=np.delete(A3,A3[:,0]<1.6,0)
A3=np.delete(A3,A3[:,0]>2.3,0)

A4=np.delete(A4,A4[:,0]<2.3,0)
A4=np.delete(A4,A4[:,0]>3,0)

A5=np.delete(A5,A5[:,0]<3.0,0)
A5=np.delete(A5,A5[:,0]>3.8,0)

A6=np.delete(A6,A6[:,0]<3.8,0)
A6=np.delete(A6,A6[:,0]>4.6,0)

AA=np.vstack([A1,A2,A3,A4,A5,A6])
_, indices = np.unique(AA[:, 0], return_index=True)
AAB=AA[indices, :]
lenght=int(len(AAB)/100)
time=np.linspace(0,np.max(AAB[:,0]),lenght*100)
Data=np.interp(time,AAB[:,0],AAB[:,1])

plt.plot(time[100*2:100*3],Data[100*2:100*3])


np.save('time',time)
np.save('Data',Data)

