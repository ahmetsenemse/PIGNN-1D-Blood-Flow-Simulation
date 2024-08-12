from tensorflow.keras import backend as K
from Gradient import *

def Physics(A,V,M_nor,U_nor,M2_nor,p0_nor,Beta_nor,A0_nor,dx_nor,pext,dt,Nodes,Kr):
        P=(pext+Beta_nor*(K.sqrt(A*M2_nor)-K.sqrt(A0_nor)))/p0_nor
        
        A_t=gradient_time(A,0,M_nor,U_nor,dt)
        v_t=gradient_time(V,0,M_nor,U_nor,dt)
        
        A_x=gradient_space(A,M_nor,dx_nor,Nodes)
        p_x=gradient_space(P,M_nor,dx_nor,Nodes)
        v_x=gradient_space(V,M_nor,dx_nor,Nodes)


        loss1=K.mean(K.square(A_t+A*v_x+V*A_x))
        loss2=K.mean(K.square(v_t+1.33*V*v_x+p_x-(Kr/U_nor/M_nor)*(V)/(A)))
        
        loss=loss1+loss2
        return loss,P
	



def bif(P1,V1,A1,P2,V2,A2,P3,V3,A3,P_1,V_1,A_1,P_2,V_2,A_2,P_3,V_3,A_3):
	Pressureloss=P1*P_1+0.5*1060*(V1*V_1)**2
	Au_loss=V1*V_1*A1*A_1

	loss1=K.mean(K.square((Pressureloss-P2*P_2-0.5*1060*(V2*V_2)**2)/P_1))
	loss2=K.mean(K.square((Pressureloss-P3*P_3-0.5*1060*(V3*V_3)**2)/P_1))
	loss3=K.mean(K.square((Au_loss-V2*V_2*A2*A_2-V3*V_3*A3*A_3)/A_1/V_1))
	loss1+loss2+loss3    
	return loss1+loss2+loss3  