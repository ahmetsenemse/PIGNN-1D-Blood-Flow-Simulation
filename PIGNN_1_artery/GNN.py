import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

import sys
sys.path.insert(1, '../Utils')

import numpy as np
import tensorflow as tf
from tensorflow import keras 
import spektral
from spektral.data import Dataset, Graph
from spektral.data.loaders import SingleLoader

from keras import backend as K
from Physics import Physics
from Model import GNN
from Train_model import fit_model

inlet_points=[0]
outlet_points=[1]

DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)
rho=1060
vis=0.0025
pext=9.986e+03
Kr=-2.0*1.33*vis*np.pi/rho/(1.33-1)
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
	Velocity=wave[:,3:4]
	Area=wave[:,5:6]
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

A_max=np.max(np.array([np.max(A0_1)]))
A_min=np.min(np.array([np.min(A0_1)]))
Beta_max=np.max(np.array([np.max(Beta_1)]))
Beta_min=np.min(np.array([np.min(Beta_1)]))
dx_max=np.max(np.array([np.max(dx_1)]))

Test_data_V=tf.stack([V_inlet_1/U_1,V_outlet_1/U_1])
Test_data_A=tf.stack([A_inlet_1/M2_1,A_outlet_1/M2_1])


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


			y=np.zeros([Nodes*len(inlet_points),N_t*2])
			y[Nodes*0,:N_t]=V_inlet_1[:,0]/U_1
			y[Nodes-1,:N_t]=V_outlet_1[:,0]/U_1
            
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
    		loss_pinn=loss_1
        
		#Measurement
    		loss_V1=K.mean(K.square(V[0][0,:]-V_true[0][0,:]))
    		loss_V2=K.mean(K.square(V[0][-1,:]-V_true[0][-1,:]))
    		loss_A1=K.mean(K.square(A[0][0,0]-A0_1[0,0]/M2_1*1.02))
    		loss_BC=loss_V1+loss_V2+loss_A1
        
    		loss=loss_pinn+loss_BC
        
    		return loss
	return loss_fn

loader  = SingleLoader(dataset) 

       
            

fit_model(loader,model,inlet_points,Test_data_A,Test_data_V,Nodes,N_t,customloss,30000)



p=model([dataset[0].x,dataset[0].a], training=False)

A=tf.exp(p[:,N_t:])
V=p[:,:N_t]

np.save('A',A*M2_1)
np.save('V',V*U_1)



