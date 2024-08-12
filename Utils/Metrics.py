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
from Physics import *

def R2(true,pred):
	sum_squares_residuals = tf.math.reduce_sum((true - pred) ** 2)
	sum_squares = tf.math.reduce_sum((true - tf.reduce_mean(true)) ** 2)
	R2 = 1 - sum_squares_residuals / sum_squares
	return R2

def R2_Lumen_Area_mt(loader,model,inlet_points,Test_data,Nodes,N_t):
	def R2_Lumen_Area(y_true,y_pred):
	    		
		A=[]
		V=[]
		V_true=[]
		for i in range(0,len(inlet_points)):
    			V.append(y_pred[Nodes*(i):Nodes*(i+1),:N_t])
    			A.append(tf.exp(y_pred[Nodes*(i):Nodes*(i+1),N_t:]))
    			V_true.append(y_true[Nodes*(i):Nodes*(i+1),:N_t])
		
		if len(inlet_points)==1:
    			R2_1=R2(Test_data[0,:,0],A[0][0])+R2(Test_data[1,:,0],A[0][-1])
		elif len(inlet_points)>1:
    			R2_1=0
    			for j in range(0,len(inlet_points)):
                		if j==0:
                        		R2_1=R2_1+R2(Test_data[j,:,0],A[j][-1])
                		else:                                
                        		R2_1=R2_1+R2(Test_data[j,:,0],A[j][0])
		return R2_1/len(inlet_points)
	return R2_Lumen_Area
	
def R2_Velocity_mt(loader,model,inlet_points,Test_data,Nodes,N_t):
	def R2_Velocity(y_true,y_pred):
	    		
		A=[]
		V=[]
		V_true=[]
		for i in range(0,len(inlet_points)):
    			V.append(y_pred[Nodes*(i):Nodes*(i+1),:N_t])
    			A.append(tf.exp(y_pred[Nodes*(i):Nodes*(i+1),N_t:]))
    			V_true.append(y_true[Nodes*(i):Nodes*(i+1),:N_t])
		
		if len(inlet_points)==1:
    			R2_1=R2(Test_data[0,:,0],V[0][0])+R2(Test_data[1,:,0],V[0][-1])
		elif len(inlet_points)>1:
    			R2_1=0
    			for j in range(0,len(inlet_points)):
                		if j==0:
                        		R2_1=R2_1+R2(Test_data[j,:,0],V[j][-1])
                		else:                                
                        		R2_1=R2_1+R2(Test_data[j,:,0],V[j][0])
		return R2_1/len(inlet_points)
	return R2_Velocity


def RRMSE(true, pred):
    num = tf.math.reduce_sum((true - pred)**2)
    den = tf.math.reduce_sum((pred)**2)
    squared_error = num/den
    rrmse_loss = (squared_error)**0.5
    return rrmse_loss

def RRMSE_Lumen_Area_mt(loader,model,inlet_points,Test_data,Nodes,N_t):    
	def RRMSE_Lumen_Area(y_true,y_pred):
		A=[]
		V=[]
		V_true=[]
		for i in range(0,len(inlet_points)):
    			V.append(y_pred[Nodes*(i):Nodes*(i+1),:N_t])
    			A.append(tf.exp(y_pred[Nodes*(i):Nodes*(i+1),N_t:]))
    			V_true.append(y_true[Nodes*(i):Nodes*(i+1),:N_t])
		
			
		RRMSE1=RRMSE(Test_data[0,:,0],A[0][0])
		if len(inlet_points)==1:
    			RRMSE1=RRMSE(Test_data[0,:,0],A[0][0])+RRMSE(Test_data[1,:,0],A[0][-1])
		elif len(inlet_points)>1:
    			RRMSE1=0
    			for j in range(0,len(inlet_points)):
                		if j==0:
                        		RRMSE1=RRMSE1+RRMSE(Test_data[j,:,0],A[j][-1])
                		else:                                
                        		RRMSE1=RRMSE1+RRMSE(Test_data[j,:,0],A[j][0])

		
		return RRMSE1/len(inlet_points)
	return RRMSE_Lumen_Area
	

def RRMSE_Velocity_mt(loader,model,inlet_points,Test_data,Nodes,N_t):    
	def RRMSE_Velocity(y_true,y_pred):
		A=[]
		V=[]
		V_true=[]
		for i in range(0,len(inlet_points)):
    			V.append(y_pred[Nodes*(i):Nodes*(i+1),:N_t])
    			A.append(tf.exp(y_pred[Nodes*(i):Nodes*(i+1),N_t:]))
    			V_true.append(y_true[Nodes*(i):Nodes*(i+1),:N_t])
		
			
		RRMSE1=RRMSE(Test_data[0,:,0],A[0][0])
		if len(inlet_points)==1:
    			RRMSE1=RRMSE(Test_data[0,:,0],A[0][0])+RRMSE(Test_data[1,:,0],V[0][-1])
		elif len(inlet_points)>1:
    			RRMSE1=0
    			for j in range(0,len(inlet_points)):
                		if j==0:
                        		RRMSE1=RRMSE1+RRMSE(Test_data[j,:,0],V[j][-1])
                		else:                                
                        		RRMSE1=RRMSE1+RRMSE(Test_data[j,:,0],V[j][0])
		
		return RRMSE1/len(inlet_points)
	return RRMSE_Velocity 
    
