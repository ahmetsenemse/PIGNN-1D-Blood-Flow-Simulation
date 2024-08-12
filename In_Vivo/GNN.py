import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dropout, Input,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,Adamax,Adagrad,Adadelta
import spektral as spk
from spektral.data import Dataset, Graph
from spektral.data.loaders import SingleLoader,BatchLoader,PackedBatchLoader
from scipy import integrate
from spektral.layers import ECCConv, GlobalSumPool, GraphMasking,CrystalConv,ARMAConv
from spektral.transforms import NormalizeAdj

#import data_creator as dat
from keras import backend as K
from tensorflow_addons.optimizers import CyclicalLearningRate


inlet_points=[0,1,1]
outlet_points=[1,2,2]

DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)
rho=1060
vis=3.5e-3
alpha=1.33
pext=10000
Nodes=10
Nt=201
Maxt=0.110

time=np.linspace(0,Maxt,Nt)
dT=time[1]-time[0]


starrt=1.115#1.522
Bp2=60/575
outlet2_data=np.load('internal_artery_outlet/data.npy')
outlet2_time=np.load('internal_artery_outlet/time.npy')
outlet2=np.stack([outlet2_time,outlet2_data])
outlet2=np.delete(outlet2,outlet2[0,:]<starrt,1)
outlet2=np.delete(outlet2,outlet2[0,:]>starrt+Bp2,1)
outlet2[0,:]=outlet2[0,:]-np.min(outlet2[0,:])
outlet2[0,:]=(outlet2[0,:])*(Maxt/np.max(outlet2[0,:]))
outlet2_data=np.interp(time,outlet2[0,:]-np.min(outlet2[0,:]),-outlet2[1,:]*1e-3)
#plt.plot(time,outlet2_data)



starrt=1.25
Bp=58/534
outlet1_data=np.load('external_artery_outlet/data.npy')
outlet1_time=np.load('external_artery_outlet/time.npy')
outlet11=np.stack([outlet1_time,outlet1_data])
outlet11=np.delete(outlet11,outlet11[0,:]<starrt,1)
outlet11=np.delete(outlet11,outlet11[0,:]>starrt+Bp,1)
outlet11[0,:]=outlet11[0,:]-np.min(outlet11[0,:])
outlet11[0,:]=(outlet11[0,:])*(Maxt/np.max(outlet11[0,:]))
outlet1_data=np.interp(time,outlet11[0,:]-np.min(outlet11[0,:]),-outlet11[1,:]*1e-3)


starrt=1.5#1.522
Bp2=73/575
inlet_data=np.load('common_artery_inlet/data.npy')
inlet_time=np.load('common_artery_inlet/time.npy')
inlet=np.stack([inlet_time,inlet_data])
inlet=np.delete(inlet,inlet[0,:]<starrt,1)
inlet=np.delete(inlet,inlet[0,:]>starrt+Bp2,1)
inlet[0,:]=inlet[0,:]-np.min(inlet[0,:])
inlet[0,:]=(inlet[0,:])*(Maxt/np.max(inlet[0,:]))
inlet_data=np.interp(time,inlet[0,:]-np.min(inlet[0,:]),-inlet[1,:]*1e-3)
    
U_1=tf.convert_to_tensor(10,dtype='float64')
p0_1=rho*U_1**2

A0_1=np.ones([Nodes,1])*0.146e-6#*0.146e-6
A0_1=tf.convert_to_tensor(A0_1,dtype='float64')
M2_1=tf.math.reduce_min(A0_1)
M_1=tf.math.sqrt(M2_1)
R0_1=K.sqrt(A0_1/3.14)
dx_1=0.005/Nodes

A0_2=np.ones([Nodes,1])*0.075e-6#*0.075e-6
A0_2=tf.convert_to_tensor(A0_2,dtype='float64')
M2_2=tf.math.reduce_min(A0_2)
M_2=tf.math.sqrt(M2_2)
R0_2=K.sqrt(A0_2/3.14)
dx_2=0.002/Nodes

A0_3=np.ones([Nodes,1])*0.049e-6#*0.049e-6
A0_3=tf.convert_to_tensor(A0_3,dtype='float64')
M2_3=tf.math.reduce_min(A0_3)
M_3=tf.math.sqrt(M2_3)
R0_3=K.sqrt(A0_3/3.14)
dx_3=0.0006/Nodes



k1=1e6
k2=3
k3=1e5

Eh1=0.1*(k1*tf.math.exp(-k2*100*R0_1)+k3)*R0_1
Beta_1=(4/3)*1.772*(Eh1/A0_1)

Eh2=0.1*(k1*tf.math.exp(-k2*100*R0_2)+k3)*R0_2
Beta_2=(4/3)*1.772*(Eh2/A0_2)

Eh3=0.1*(k1*tf.math.exp(-k2*100*R0_3)+k3)*R0_3
Beta_3=(4/3)*1.772*(Eh3/A0_3)

A_max=np.max(np.array([np.max(A0_1),np.max(A0_2),np.max(A0_3)]))
A_min=np.min(np.array([np.min(A0_1),np.min(A0_2),np.min(A0_3)]))
Beta_max=np.max(np.array([np.max(Beta_1),np.max(Beta_2),np.max(Beta_3)]))
Beta_min=np.min(np.array([np.min(Beta_1),np.min(Beta_2),np.min(Beta_3)]))
dx_max=np.max(np.array([np.max(dx_1),np.max(dx_2),np.max(dx_3)]))    
   




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


            a=spk.utils.convolution.normalized_laplacian(a, symmetric=True)
            a=spk.utils.convolution.rescale_laplacian(a, lmax=None)



            x=np.zeros([Nodes*len(inlet_points),3])
            for i in range(0,Nodes):
                x[i,0]=(A0_1[i,0]-A_min)/(A_max-A_min)
                x[i,1]=(Beta_1[i]-Beta_min)/(Beta_max-Beta_min)
                x[i,2]=dx_1/dx_max
                x[i+10,0]=(A0_2[i,0]-A_min)/(A_max-A_min)
                x[i+10,1]=(Beta_2[i]-Beta_min)/(Beta_max-Beta_min)
                x[i+10,2]=dx_2/dx_max
                x[i+20,0]=(A0_3[i,0]-A_min)/(A_max-A_min)
                x[i+20,1]=(Beta_3[i]-Beta_min)/(Beta_max-Beta_min)
                x[i+20,2]=dx_3/dx_max

                            
            y=np.zeros([Nodes*3,Nt*2])
            y[Nodes*2-1,:Nt]=outlet1_data/U_1
            y[Nodes*3-1,:Nt]=outlet2_data/U_1
            
            return Graph(x=x, a=a,y=y)

        # We must return a list of Graph objects
        return [make_graph(i) for i in range(self.n_samples)]    
    
    
dataset = MyDataset(1)


N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes


acti2='LeakyReLU'
acti='tanh'

order_number=1
channel_number=32
it_number=1
# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True)

gc_1 = ARMAConv(
    channel_number,
    iterations=it_number,
    order=order_number,
    share_weights=True,
    activation=acti,
    gcn_activation=acti2,
)([x_in, a_in])

for i in range(0,4):
    gc_1 = ARMAConv(
        channel_number,
        iterations=it_number,
        order=order_number,
        share_weights=True,
        activation=acti,
        gcn_activation=acti2,
    )([gc_1, a_in])
gc_2 = ARMAConv(
    n_out,
    iterations=it_number,
    order=order_number,
    share_weights=True,
    activation=acti,
    gcn_activation=None,
)([gc_1, a_in])



model = Model(inputs=[x_in, a_in], outputs=gc_2)


@tf.function
def gradinet_time(Y,N_type,M,U):
    
	if N_type==1:
		T=1
		Y_t1=(Y[:,1:2]-Y[:,0:1])/(dT)
		Y_t2=(Y[:,2:]-Y[:,:-2])/(dT*2)
		Y_t3=(Y[:,-2:-1]-Y[:,-3:-2])/(dT)
	else:
		T=M/U
		Y_t1=(Y[:,1:2]-Y[:,0:1])/(dT/T)
		Y_t2=(Y[:,2:]-Y[:,:-2])/(2*dT/T)
		Y_t3=(Y[:,-2:-1]-Y[:,-3:-2])/(dT/T)


	Y_t=tf.concat([Y_t1, Y_t2, Y_t3],axis=1)
	return Y_t

@tf.function
def gradinet_space(Y,M,dx):

	Y_x1=(Y[1:2,:]-Y[0:1,:])/(dx/M)
	Y_x2=(Y[2:,:]-Y[:-2,:])/(2*dx/M)
	Y_x3=(Y[-2:-1,:]-Y[-3:-2,:])/(dx/M)

	Y_x=tf.concat([Y_x1, Y_x2, Y_x3],axis=0)
    
	return Y_x

def PINN(A,V,V_true,M_nor,U_nor,M2_nor,p0_nor,Beta_nor,A0_nor,dx_nor):
        P=(pext+Beta_nor*(K.sqrt(A*M2_nor)-K.sqrt(A0_nor)))/p0_nor
        
        A_t=gradinet_time(A,0,M_nor,U_nor)
        v_t=gradinet_time(V,0,M_nor,U_nor)
        
        A_x=gradinet_space(A,M_nor,dx_nor)
        p_x=gradinet_space(P,M_nor,dx_nor)
        v_x=gradinet_space(V,M_nor,dx_nor)
        

        Kr=-2.0*alpha*vis*np.pi/rho/(alpha-1)

        loss1=K.mean(K.square(A_t+A*v_x+V*A_x))
        loss2=K.mean(K.square(v_t+alpha*V*v_x+p_x-(Kr/U_nor/M_nor)*(V)/(A)))
        
        loss=loss1+loss2
        return loss,P



def customloss():
    def loss_fn(y_true,y_pred):
        V_1=y_pred[:Nodes,:Nt]
        V_true_1=y_true[:Nodes,:Nt]
        A_1=tf.exp(y_pred[:Nodes,Nt:])

        V_2=y_pred[Nodes:(Nodes)*2,:Nt]
        V_true_2=y_true[Nodes:(Nodes)*2,:Nt]
        A_2=tf.exp(y_pred[Nodes:(Nodes)*2,Nt:])
        
        V_3=y_pred[(Nodes)*2:,:Nt]
        V_true_3=y_true[(Nodes)*2:,:Nt]
        A_3=tf.exp(y_pred[(Nodes)*2:,Nt:])
        
        #PINN
        loss_1,P_1=PINN(A_1,V_1,V_true_1,M_1,U_1,M2_1,p0_1,Beta_1,A0_1,dx_1)
        loss_2,P_2=PINN(A_2,V_2,V_true_2,M_2,U_1,M2_2,p0_1,Beta_2,A0_2,dx_2)
        loss_3,P_3=PINN(A_3,V_3,V_true_3,M_3,U_1,M2_3,p0_1,Beta_3,A0_3,dx_3)
        
        #MC
        #lossV1=K.mean(K.square(V_1[0,:]-V_true_1[0,:]))
        lossV2=K.mean(K.square(V_2[-1,:]-V_true_2[-1,:]))    
        lossV1=K.mean(K.square(V_1[0,:]-inlet_data/U_1))
        lossV3=K.mean(K.square(V_3[-1,:]-V_true_3[-1,:]))  

        
        #IC
        lossA1=K.mean(K.square(A_1[:,0]-A_1[:,-1]))
        lossA2=K.mean(K.square(A_2[:,0]-A_2[:,-1]))
        lossA3=K.mean(K.square(A_3[:,0]-A_3[:,-1]))
        lossA4=K.mean(K.square(A_1[:,0]-1))
        lossA5=K.mean(K.square(A_2[:,0]-1))
        lossA6=K.mean(K.square(A_3[:,0]-1))
        
        #loss_interaction
        
        Pressureloss=P_1[-1,:]*p0_1+0.5*rho*(V_1[-1,:]*U_1)**2
        Au_loss=V_1[-1,:]*U_1*A_1[-1,:]*M2_1

        
        loss1=K.mean(K.square((Pressureloss-P_2[0,:]*p0_1-0.5*rho*(V_2[0,:]*U_1)**2)/p0_1))
        loss2=K.mean(K.square((Pressureloss-P_3[0,:]*p0_1-0.5*rho*(V_3[0,:]*U_1)**2)/p0_1))
        loss3=K.mean(K.square((Au_loss-V_2[0,:]*U_1*A_2[0,:]*M2_2-V_3[0,:]*U_1*A_3[0,:]*M2_3)/M2_1/U_1))
        
        lossPINN=loss_1+loss_2+loss_3
        lossMC=lossV2+lossV3+lossV1
        lossAC=lossA5+lossA6+lossA4+lossA2+lossA3+lossA1
        lossBIF=loss1+loss2+loss3
        
        loss=lossPINN+lossMC+lossAC+lossBIF
        return loss
    return loss_fn

loader  = SingleLoader(dataset) 

optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss=customloss(),  # To compute mean
)

def scheduler(epoch, lr):
    lr=0.001
    if epoch>90000:
    	lr=0.0001
    return lr
        

model.fit(
    loader.load(),
    steps_per_epoch=1,
    epochs=100000,
    verbose=2
)


i=0
for batch in loader:
    inputs, target = batch
    p = model(inputs, training=False)
    i=i+1
    if i==2:
        break
    
V_1=p[:Nodes,:Nt]*U_1
A_1=tf.exp(p[:Nodes,Nt:])*M2_1
Q_1=A_1*V_1

V_2=p[Nodes:(Nodes)*2,:Nt]*U_1
A_2=tf.exp(p[Nodes:(Nodes)*2,Nt:])*M2_2
Q_2=A_2*V_2

V_3=p[(Nodes)*2:,:Nt]*U_1
A_3=tf.exp(p[(Nodes)*2:,Nt:])*M2_3
Q_3=A_3*V_3   


np.save('common_V',V_1)
np.save('common_A',A_1)
np.save('common_Q',Q_1)

np.save('external_V',V_2)
np.save('external_A',A_2)
np.save('external_Q',Q_2)

np.save('internal_V',V_3)
np.save('internal_A',A_3)
np.save('internal_Q',Q_3)

inlet_data1=np.load('needed.npy')

starrt=2.855#1.522
Bp2=60/575
inlet_data=np.load('common_artery_Bif/data.npy')
inlet_time=np.load('common_artery_Bif/time.npy')
inlet=np.stack([inlet_time,inlet_data])
inlet=np.delete(inlet,inlet[0,:]<starrt,1)
inlet=np.delete(inlet,inlet[0,:]>starrt+Bp2,1)
inlet[0,:]=inlet[0,:]-np.min(inlet[0,:])
inlet[0,:]=(inlet[0,:])*(Maxt/np.max(inlet[0,:]))
inlet_data=np.interp(time,inlet[0,:]-np.min(inlet[0,:]),-inlet[1,:]*1e-3)

plt.plot(time,inlet_data,'b')
#plt.plot(time[:],inlet_data1)
plt.plot(time[:],A_1[0,:],'r')
plt.plot(time[:],V_1[-1,:],'r')

np.save('bifurcation',inlet_data)
