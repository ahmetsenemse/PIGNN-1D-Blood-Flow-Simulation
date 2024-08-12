import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

import numpy as np
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import spektral
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,Adagrad,Adamax
from spektral.data import Dataset, Graph
from spektral.data.loaders import SingleLoader
from spektral.layers import ARMAConv
from spektral.transforms import NormalizeAdj
import datetime

from keras import backend as K
from Physics import Physics, bif
from Model import GNN
from Train_model import fit_model

inlet_points=[0,1,1,2,2,3,3]
outlet_points=[1,2,3,4,5,6,7]

DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)
rho=1060
vis=0.0025
alpha=1.33
pext=9.986e+03
Kr=-2.0*alpha*vis*np.pi/rho/(alpha-1)
Nodes=10


def wave_input(name,location,tmin,tmax):
	wave=np.loadtxt(name,delimiter=' ')
	if location==1:
		wave=np.delete(wave,wave[:,6]!=np.max(wave[:,6]),0)
	else:
		wave=np.delete(wave,wave[:,6]!=1,0)
		
	wave=np.delete(wave,wave[:,0]<tmin,0)
	wave=np.delete(wave,wave[:,0]>tmax,0)	
	
	wave=np.delete(wave, list(range(1, wave.shape[0], 2)), axis=0)
	Velocity=wave[:,2:3]
	Area=wave[:,4:5]
	Time=wave[:,0:1]
	return	Velocity,Area,Time

def Mechanical_prop(lenght,limit1,limit2):
	dx=np.linspace(0,lenght,Nodes)
	R0=(limit1*dx+limit2)
	A0=3.14* (limit1*dx+limit2) ** 2
	Eh=0.1*(3000000*tf.math.exp(-9*100*R0)+489871.1841)*R0
	Beta=(4/3)*1.772*(Eh/A0)
	A0=tf.reshape(A0,[Nodes,1])
	Beta=tf.reshape(Beta,[Nodes,1])
	M2=tf.math.reduce_min(A0)
	M=tf.math.sqrt(M2)
	U=((2*tf.math.reduce_max(Eh))/(3*rho*tf.math.reduce_min(R0)))**0.5
	p0=rho*U**2
	
	return dx,A0,Beta,M2,M,U,p0
	

V_inlet_1,A_inlet_1,t=wave_input("Snaps/sim_1_1.his",0,4.94,5.76)
V_outlet_1,A_outlet_1,t=wave_input("Snaps/sim_1_1.his",1,4.94,5.76)

dx_1,A0_1,Beta_1,M2_1,M_1,U_1,p0_1=Mechanical_prop(0.06,-0.0033667,0.018206)
dt=t[1]-t[0]
N_t=len(t)
mini=np.min(A_inlet_1)

V_inlet_2,A_inlet_2,t=wave_input("Snaps/sim_1_2.his",0,4.94,5.76)
V_outlet_2,A_outlet_2,t=wave_input("Snaps/sim_1_2.his",1,4.94,5.76)

dx_2,A0_2,Beta_2,M2_2,M_2,U_2,p0_2=Mechanical_prop(0.02,0,0.015799)

V_inlet_3,A_inlet_3,t=wave_input("Snaps/sim_1_3.his",0,4.94,5.76)
V_outlet_3,A_outlet_3,t=wave_input("Snaps/sim_1_3.his",1,4.94,5.76)

dx_3,A0_3,Beta_3,M2_3,M_3,U_3,p0_3=Mechanical_prop(0.034,0,0.008561)


V_inlet_4,A_inlet_4,t=wave_input("Snaps/sim_1_6.his",0,4.94,5.76)
V_outlet_4,A_outlet_4,t=wave_input("Snaps/sim_1_6.his",1,4.94,5.76)

dx_4,A0_4,Beta_4,M2_4,M_4,U_4,p0_4=Mechanical_prop(0.0390,0,0.015187)


V_inlet_5,A_inlet_5,t=wave_input("Snaps/sim_1_7.his",0,4.94,5.76)
V_outlet_5,A_outlet_5,t=wave_input("Snaps/sim_1_7.his",1,4.94,5.76)

dx_5,A0_5,Beta_5,M2_5,M_5,U_5,p0_5=Mechanical_prop(0.1390,0,0.004532)


V_inlet_6,A_inlet_6,t=wave_input("Snaps/sim_1_4.his",0,4.94,5.76)
V_outlet_6,A_outlet_6,t=wave_input("Snaps/sim_1_4.his",1,4.94,5.76)

dx_6,A0_6,Beta_6,M2_6,M_6,U_6,p0_6=Mechanical_prop(0.034,0,0.007348)


V_inlet_7,A_inlet_7,t=wave_input("Snaps/sim_1_5.his",0,4.94,5.76)
V_outlet_7,A_outlet_7,t=wave_input("Snaps/sim_1_5.his",1,4.94,5.76)

dx_7,A0_7,Beta_7,M2_7,M_7,U_7,p0_7=Mechanical_prop(0.0940,0,0.004899)

A_max=np.max(np.array([np.max(A0_1),np.max(A0_2),np.max(A0_3),np.max(A0_4),np.max(A0_5),np.max(A0_6),np.max(A0_7)]))
A_min=np.min(np.array([np.min(A0_1),np.min(A0_2),np.min(A0_3),np.min(A0_4),np.min(A0_5),np.min(A0_6),np.min(A0_7)]))
Beta_max=np.max(np.array([np.max(Beta_1),np.max(Beta_2),np.max(Beta_3),np.max(Beta_4),np.max(Beta_5),np.max(Beta_6),np.max(Beta_7)]))
Beta_min=np.min(np.array([np.min(Beta_1),np.min(Beta_2),np.min(Beta_3),np.min(Beta_4),np.min(Beta_5),np.min(Beta_6),np.min(Beta_7)]))
dx_max=np.max(np.array([np.max(dx_1),np.max(dx_2),np.max(dx_3),np.max(dx_4),np.max(dx_5),np.max(dx_6),np.max(dx_7)]))


Test_data_V=tf.stack([V_outlet_1/U_1,V_inlet_2/U_2,V_inlet_3/U_3,V_inlet_4/U_4,V_inlet_5/U_5,V_inlet_6/U_6,V_inlet_7/U_7])
Test_data_A=tf.stack([A_outlet_1/M2_1,A_inlet_2/M2_2,A_inlet_3/M2_3,A_inlet_4/M2_4,A_inlet_5/M2_5,A_inlet_6/M2_6,A_inlet_7/M2_7])

class MyDataset(Dataset):

	def __init__(self,inlet_points, **kwargs):
		self.n_samples = 1
		super().__init__(**kwargs)

	def read(self):
		def make_graph(i):
		
			a=np.zeros([Nodes*len(inlet_points),Nodes*len(inlet_points)])
			Nodes_counter=0
			for j in range(0,len(inlet_points)):
				for i in range(0,Nodes):
					if i%Nodes==0:
						a[Nodes_counter,Nodes_counter+1]=1
					elif i%Nodes==Nodes-1:
						a[Nodes_counter,Nodes_counter-1]=1
					else:
						a[Nodes_counter,Nodes_counter+1]=1
						a[Nodes_counter,Nodes_counter-1]=1
					Nodes_counter=Nodes_counter+1


			for i in range(0,len(inlet_points)):
				for j in range(0,len(inlet_points)):
					if i!=j:
						if outlet_points[i]==inlet_points[j]:
							a[(Nodes)*(i+1)-1,(Nodes)*j]=1
							a[(Nodes)*j,(Nodes)*(i+1)-1]=1
						elif inlet_points[i]==inlet_points[j]:
							a[(Nodes)*i,(Nodes)*j]=1
							a[(Nodes)*j,(Nodes)*i]=1
						elif outlet_points[i]==outlet_points[j]:
							a[(Nodes)*(i+1)-1,(Nodes)*(j+1)-1]=1
						a[(Nodes)*(j+1)-1,(Nodes)*(i+1)-1]=1


			a=spektral.utils.convolution.normalized_laplacian(a, symmetric=True)
			a=spektral.utils.convolution.rescale_laplacian(a, lmax=None)

			x=np.zeros([Nodes*len(inlet_points),3])
			for i in range(0,Nodes):
				x[i,0]=(A0_1[i,0]-A_min)/(A_max-A_min)
				x[i,1]=(Beta_1[i]-Beta_min)/(Beta_max-Beta_min)
				x[i,2]=dx_1[1]*i/dx_max
				
				x[i+10,0]=(A0_2[i,0]-A_min)/(A_max-A_min)
				x[i+10,1]=(Beta_2[i]-Beta_min)/(Beta_max-Beta_min)
				x[i+10,2]=dx_2[1]*i/dx_max
				
				x[i+20,0]=(A0_3[i,0]-A_min)/(A_max-A_min)
				x[i+20,1]=(Beta_3[i]-Beta_min)/(Beta_max-Beta_min)
				x[i+20,2]=dx_3[1]*i/dx_max
				
				x[i+30,0]=(A0_4[i,0]-A_min)/(A_max-A_min)
				x[i+30,1]=(Beta_4[i]-Beta_min)/(Beta_max-Beta_min)
				x[i+30,2]=dx_4[1]*i/dx_max
				
				x[i+40,0]=(A0_5[i,0]-A_min)/(A_max-A_min)
				x[i+40,1]=(Beta_5[i]-Beta_min)/(Beta_max-Beta_min)
				x[i+40,2]=dx_5[1]*i/dx_max
				
				x[i+50,0]=(A0_6[i,0]-A_min)/(A_max-A_min)
				x[i+50,1]=(Beta_6[i]-Beta_min)/(Beta_max-Beta_min)
				x[i+50,2]=dx_6[1]*i/dx_max
				
				x[i+60,0]=(A0_7[i,0]-A_min)/(A_max-A_min)
				x[i+60,1]=(Beta_7[i]-Beta_min)/(Beta_max-Beta_min)
				x[i+60,2]=dx_7[1]*i/dx_max

			y=np.zeros([Nodes*len(inlet_points),N_t*2])
			y[Nodes*0,:N_t]=V_inlet_1[:,0]/U_1
			y[Nodes*2-1,:N_t]=V_outlet_2[:,0]/U_2
			y[Nodes*3-1,:N_t]=V_outlet_3[:,0]/U_3
			y[Nodes*4-1,:N_t]=V_outlet_4[:,0]/U_4
			y[Nodes*5-1,:N_t]=V_outlet_5[:,0]/U_5
			y[Nodes*6-1,:N_t]=V_outlet_6[:,0]/U_6
			y[Nodes*7-1,:N_t]=V_outlet_7[:,0]/U_7
            
			return Graph(x=x, a=a,y=y)


		return [make_graph(i) for i in range(self.n_samples)]    
    
    
dataset = MyDataset(1)

N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes


model=GNN(N,F,n_out,32,1,1,'tanh','LeakyReLU')

    
def customloss():
	def loss_fn(y_true,y_pred):
    
    		A=[]
    		V=[]
    		V_true=[]
    		for i in range(0,len(inlet_points)):
    			V.append(y_pred[Nodes*(i):Nodes*(i+1),:N_t])
    			A.append(tf.exp(y_pred[Nodes*(i):Nodes*(i+1),N_t:]))
    			V_true.append(y_true[Nodes*(i):Nodes*(i+1),:N_t])
        
		#PINN
    		loss_1,P_1=Physics(A[0],V[0],M_1,U_1,M2_1,p0_1,Beta_1,A0_1,dx_1[1]-dx_1[0],pext,dt,Nodes,Kr)
    		loss_2,P_2=Physics(A[1],V[1],M_2,U_2,M2_2,p0_2,Beta_2,A0_2,dx_2[1]-dx_2[0],pext,dt,Nodes,Kr)
    		loss_3,P_3=Physics(A[2],V[2],M_3,U_3,M2_3,p0_3,Beta_3,A0_3,dx_3[1]-dx_3[0],pext,dt,Nodes,Kr)
    		loss_4,P_4=Physics(A[3],V[3],M_4,U_4,M2_4,p0_4,Beta_4,A0_4,dx_4[1]-dx_4[0],pext,dt,Nodes,Kr)
    		loss_5,P_5=Physics(A[4],V[4],M_5,U_5,M2_5,p0_5,Beta_5,A0_5,dx_5[1]-dx_5[0],pext,dt,Nodes,Kr)
    		loss_6,P_6=Physics(A[5],V[5],M_6,U_6,M2_6,p0_6,Beta_6,A0_6,dx_6[1]-dx_6[0],pext,dt,Nodes,Kr)
    		loss_7,P_7=Physics(A[6],V[6],M_7,U_7,M2_7,p0_7,Beta_7,A0_7,dx_7[1]-dx_7[0],pext,dt,Nodes,Kr)
    		loss_pinn=loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7
        
		#Measurement
    		loss_V1=K.mean(K.square(V[0][0,:]-V_true[0][0,:]))
    		#loss_V2=K.mean(K.square(V[1][-1,:]-V_true[1][-1,:]))
    		#loss_V2=K.mean(K.square(V[2][-1,:]-V_true[2][-1,:]))
    		loss_V2=K.mean(K.square(V[3][-1,:]-V_true[3][-1,:]))
    		loss_V3=K.mean(K.square(V[4][-1,:]-V_true[4][-1,:]))
    		loss_V4=K.mean(K.square(V[5][-1,:]-V_true[5][-1,:]))
    		loss_V5=K.mean(K.square(V[6][-1,:]-V_true[6][-1,:]))
    		loss_A1=K.mean(K.square(A[0][0,0]-mini/M2_1))
    		loss_A2=K.mean(K.square(y_pred[:,N_t]-y_pred[:,-1]))
    		loss_BC=loss_V1+loss_V2+loss_V3+loss_V4+loss_V5+loss_A1+loss_A2
        
		#loss_interaction
    		loss_IC1=bif(P_1[-1,:],V[0][-1,:],A[0][-1,:],P_2[0,:],V[1][0,:],A[1][0,:],P_3[0,:],V[2][0,:],A[2][0,:],p0_1,U_1,M2_1,p0_2,U_2,M2_2,p0_3,U_3,M2_3)
    		loss_IC2=bif(P_2[-1,:],V[1][-1,:],A[1][-1,:],P_4[0,:],V[3][0,:],A[3][0,:],P_5[0,:],V[4][0,:],A[4][0,:],p0_2,U_2,M2_2,p0_4,U_4,M2_4,p0_5,U_5,M2_5)
    		loss_IC3=bif(P_3[-1,:],V[2][-1,:],A[2][-1,:],P_6[0,:],V[5][0,:],A[5][0,:],P_7[0,:],V[6][0,:],A[6][0,:],p0_3,U_3,M2_3,p0_6,U_6,M2_6,p0_7,U_7,M2_7)
        	
    		loss_IC=loss_IC1+loss_IC2+loss_IC3
    		loss=loss_pinn+loss_IC+loss_BC
        
    		return loss
	return loss_fn

loader  = SingleLoader(dataset) 



fit_model(loader,model,inlet_points,Test_data_A,Test_data_V,Nodes,N_t,customloss,100000)


p=model([dataset[0].x,dataset[0].a], training=False)

A=tf.exp(p[:,N_t:])
V=p[:,:N_t]

np.save('A1',A[:10,:]*M2_1)
np.save('V1',V[:10,:]*U_1)

np.save('A2',A[10:20,:]*M2_2)
np.save('V2',V[10:20,:]*U_2)

np.save('A3',A[20:30,:]*M2_3)
np.save('V3',V[20:30,:]*U_3)

np.save('A4',A[30:40,:]*M2_4)
np.save('V4',V[30:40,:]*U_4)

np.save('A5',A[40:50,:]*M2_5)
np.save('V5',V[40:50,:]*U_5)

np.save('A6',A[50:60,:]*M2_6)
np.save('V6',V[50:60,:]*U_6)

np.save('A7',A[60:,:]*M2_7)
np.save('V7',V[60:,:]*U_7)





