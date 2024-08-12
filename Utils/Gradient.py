import tensorflow as tf

@tf.function
def gradient_time(Y,N_type,M,U,dt):
    
	if N_type==1: #gradient
		T=1
		Y_t1=(Y[:,1:2]-Y[:,-2:-1])/(dt*2)
		Y_t2=(Y[:,2:]-Y[:,:-2])/(dt*2)
		Y_t3=(Y[:,1:2]-Y[:,-2:-1])/(dt*2)
	else:	     #adimensionalized gradient
		T=M/U
		Y_t1=(Y[:,1:2]-Y[:,-2:-1])/(dt*2/T)
		Y_t2=(Y[:,2:]-Y[:,:-2])/(dt*2/T)
		Y_t3=(Y[:,1:2]-Y[:,-2:-1])/(dt*2/T)


	Y_t=tf.concat([Y_t1, Y_t2, Y_t3],axis=1)
	return Y_t


@tf.function
def gradient_space(Y,M,dx,Nodes):

	Y_x1=(Y[1:2,:]-Y[0:1,:])/(dx/M)
	Y_x2=(Y[2:,:]-Y[:-2,:])/(2*dx/M)
	Y_x3=(Y[Nodes-1:Nodes,:]-Y[Nodes-2:Nodes-1,:])/(dx/M)

	Y_x=tf.concat([Y_x1, Y_x2, Y_x3],axis=0)
    
	return Y_x




