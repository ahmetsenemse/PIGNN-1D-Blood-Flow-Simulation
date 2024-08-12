import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(name,name1,Fig_name,Xaxis,Yaxis):
	plt.rcParams["figure.autolayout"] = True
	
	Nektar1D=np.load(name1)
	Tn=np.linspace(0,30, num=len(Nektar1D))*0.6
	
	df = pd.read_csv(name)
	TSBOARD_SMOOTHING = 0.90

	smooth = []
	smooth.append(df.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	    
	plt.yscale('log')
	plt.plot(Tn,Nektar1D,'r-', linewidth=3)
	plt.plot(df['Step']*1e-3,smooth[0]["Value"],'b-', linewidth=3)

	
	#plt.xlabel(Xaxis,fontsize=20)
	plt.ylabel(Yaxis,fontsize=20)
	plt.grid(alpha=0.3)
	plt.ylim((0.0001,1.05))
	plt.xlim((0,30))
	
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=15)
	#plt.tick_params(
    	#	axis='x',          # changes apply to the x-axis
    	#	which='both',      # both major and minor ticks are affected
    	#	bottom=False,      # ticks along the bottom edge are off
    	#	top=False,         # ticks along the top edge are off
    	#	labelbottom=False)
	plt.tick_params(left = True, right = True , labelleft = True , 
                labelbottom = False, bottom = False) 
	plt.legend(['Nektar1D', 'PIGNN'],prop = { "size": 16})
	"""
	newax = ax.twiny()
	newax.set_frame_on(True)
	newax.patch.set_visible(False)
	newax.xaxis.set_ticks_position('bottom')
	newax.xaxis.set_label_position('bottom')
	newax.spines['bottom'].set_position(('outward', 60))
	newax.set_xticks((0,7.5,15,22.5,30,37.5,45))
	newax.tick_params(axis='x', which='major', labelsize=15)
	newax.set_xlabel("Time (s)",fontsize=20)
	"""
			
	plt.savefig(Fig_name)
	plt.cla()
	plt.clf()    


#plot("logs/Area_metric.csv",'Snaps/r2a.npy','A_logs','Epochs (x$10^3$)','$R^2$')
#plot("logs/Velocity_metric.csv",'Snaps/r2v.npy','V_logs','Epochs (x$10^3$)','$R^2$')
plot("logs/Area_RMSE.csv",'Snaps/rmsea.npy','AR_logs','Epochs (x$10^3$)','$RRMSE$ (log)')
#plot("logs/Velocity_RMSE.csv",'Snaps/rmsev.npy','VR_logs','Epochs (x$10^3$)','$RRMSE$ (log)')


