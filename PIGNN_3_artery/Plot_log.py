import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(name1,name2,name3,Fig_name,Xaxis,Yaxis):
	plt.rcParams["figure.autolayout"] = True
	df1 = pd.read_csv(name1)
	df2 = pd.read_csv(name2)
	df3 = pd.read_csv(name3)
	TSBOARD_SMOOTHING = 0.90

	smooth = []
	smooth.append(df1.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df2.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df3.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())

	plt.yscale('log')    
	plt.plot(df1['Step']*1e-3,smooth[0]["Value"],'b-', linewidth=3,label='$PIGNNs_u$')
	plt.plot(df2['Step']*1e-3,smooth[1]["Value"],'g-', linewidth=3,label='$PINNs_{u,A}$')
	plt.plot(df3['Step']*1e-3*3,smooth[2]["Value"],'k-', linewidth=3,label='$PINNs_u$')
	#plt.legend(prop = { "size": 16})
	#plt.xlabel(Xaxis,fontsize=20)
	#plt.ylabel(Yaxis,fontsize=20)
	plt.grid(alpha=0.3)
	plt.ylim((0.001,1))
	plt.xlim((0,300))
	#plt.tick_params(
    	#	axis='x',          # changes apply to the x-axis
    	#	which='both',      # both major and minor ticks are affected
    	#	bottom=False,      # ticks along the bottom edge are off
    	#	top=False,         # ticks along the top edge are off
    	#	labelbottom=False)
	plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
                
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=15)
	
	"""
	newax = ax.twiny()
	newax.set_frame_on(True)
	newax.patch.set_visible(False)
	newax.xaxis.set_ticks_position('bottom')
	newax.xaxis.set_label_position('bottom')
	newax.spines['bottom'].set_position(('outward', 60))
	newax.set_xticks((0,25,50,75,100,125,150))
	newax.tick_params(axis='both', which='major', labelsize=15)
	newax.set_xlabel("PINN time (min)",fontsize=20)
	
	newax2 = ax.twiny()
	newax2.set_frame_on(True)
	newax2.patch.set_visible(False)
	newax2.xaxis.set_ticks_position('bottom')
	newax2.xaxis.set_label_position('bottom')
	newax2.spines['bottom'].set_position(('outward', 120))
	newax2.set_xticks((0,2,4,6,8,10,12))
	newax2.tick_params(axis='both', which='major', labelsize=15)
	newax2.set_xlabel("PIGNN time (min)",fontsize=20)
	"""
	
	plt.savefig(Fig_name)
	plt.cla()
	plt.clf()    


#plot("logs/Area_metric.csv",'../ClassicalPINN/logs/lossfn1_metric.csv','../ClassicalPINN_A/logs/lossfn1_metric.csv','A_logs','Epochs (x$10^3$)','$R^2$')
#plot("logs/Velocity_metric.csv",'../ClassicalPINN/logs/lossfn2_metric.csv','../ClassicalPINN_A/logs/lossfn2_metric.csv','V_logs','Epochs (x$10^3$)','$R^2$')

#plot("logs/lossfn3_metric.csv",'../ClassicalPINN/logs/lossfn3_metric.csv','../ClassicalPINN_A/logs/lossfn3_metric.csv','AR_logs','Epochs (x$10^3$)','$RRMSE$ (log)')
plot("logs/lossfn4_metric.csv",'../ClassicalPINN/logs/lossfn4_metric.csv','../ClassicalPINN_A/logs/lossfn4_metric.csv','VR_logs','Epochs (x$10^3$)','$RRMSE$')

