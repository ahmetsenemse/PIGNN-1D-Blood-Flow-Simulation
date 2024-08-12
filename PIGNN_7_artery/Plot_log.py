import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(name1,name2,name3,name4,name5,name6,name7,name8,Fig_name,Xaxis,Yaxis):
	plt.rcParams["figure.autolayout"] = True
	df1 = pd.read_csv(name1)
	df2 = pd.read_csv(name2)
	df3 = pd.read_csv(name3)
	df4 = pd.read_csv(name4)
	df5 = pd.read_csv(name5)
	df6 = pd.read_csv(name6)
	df7 = pd.read_csv(name7)
	df8 = pd.read_csv(name8)
	TSBOARD_SMOOTHING = 0.90

	smooth = []
	smooth.append(df1.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df2.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df3.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df4.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df5.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df6.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df7.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
	smooth.append(df8.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean())
		    
	plt.yscale('log')
	plt.plot(df1['Step']*1e-3,smooth[0]["Value"],'b-', linewidth=3,label='$Scenario-1$')
	plt.plot(df2['Step']*1e-3,smooth[1]["Value"],'g-', linewidth=3,label='$Scenario-2$')
	plt.plot(df3['Step']*1e-3,smooth[2]["Value"],'k-', linewidth=3,label='$Scenario-3$')
	plt.plot(df4['Step']*1e-3,smooth[3]["Value"],'c-', linewidth=3,label='$Scenario-4$')
	plt.plot(df5['Step']*1e-3,smooth[4]["Value"],'m-', linewidth=3,label='$Scenario-5$')
	plt.plot(df6['Step']*1e-3,smooth[5]["Value"],'y-', linewidth=3,label='$Scenario-6$')
	plt.plot(df7['Step']*1e-3,smooth[6]["Value"],'tab:orange', linewidth=3,label='$Scenario-7$')
	plt.plot(df8['Step']*1e-3,smooth[7]["Value"],'tab:gray', linewidth=3,label='$Scenario-8$')
	
	#plt.legend(prop = { "size": 16})
	

	#plt.xlabel(Xaxis,fontsize=20)
	plt.ylabel(Yaxis,fontsize=20)
	plt.grid(alpha=0.3)
	plt.ylim((0.001,5))
	plt.xlim((0,80))
	
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=15)
	#plt.tick_params(
    	#	axis='x',          # changes apply to the x-axis
    	#	which='both',      # both major and minor ticks are affected
    	#	bottom=False,      # ticks along the bottom edge are off
    	#	top=False,         # ticks along the top edge are off
    	#	labelbottom=False)
	plt.tick_params(left = True, right = False , labelleft = True , 
                labelbottom = False, bottom = False)
    		
	#ax.legend(bbox_to_anchor=(1.1, 1.05))
	"""
	newax = ax.twiny()
	newax.set_frame_on(True)
	newax.patch.set_visible(False)
	newax.xaxis.set_ticks_position('bottom')
	newax.xaxis.set_label_position('bottom')
	newax.spines['bottom'].set_position(('outward', 60))
	newax.set_xticks((0,1,2,3,4,5,6,7,8))
	newax.tick_params(axis='both', which='major', labelsize=15)
	newax.set_xlabel("Time (min)",fontsize=20)
	"""
	
	plt.savefig(Fig_name)
	plt.cla()
	plt.clf()    


#plot("logs/Area_metric.csv",'../PIGNN_7_artery_1/logs/Area_metric.csv','../PIGNN_7_artery_2/logs/Area_metric.csv','../PIGNN_7_artery_3/logs/Area_metric.csv','../PIGNN_7_artery_4/logs/Area_metric.csv','../PIGNN_7_artery_5/logs/Area_metric.csv','../PIGNN_7_artery_7/logs/lossfn1_metric.csv','../PIGNN_7_artery_8/logs/lossfn1_metric.csv','A_logs','Epochs (x$10^3$)','$R^2$')
#plot("logs/Velocity_metric.csv",'../PIGNN_7_artery_1/logs/Velocity_metric.csv','../PIGNN_7_artery_2/logs/Velocity_metric.csv','../PIGNN_7_artery_3/logs/Velocity_metric.csv','../PIGNN_7_artery_4/logs/Velocity_metric.csv','../PIGNN_7_artery_5/logs/Velocity_metric.csv','../PIGNN_7_artery_7/logs/lossfn2_metric.csv','../PIGNN_7_artery_8/logs/lossfn2_metric.csv','V_logs','Epochs (x$10^3$)','$R^2$')


plot("logs/lossfn3_metric.csv",'../PIGNN_7_artery_1/logs/lossfn3_metric.csv','../PIGNN_7_artery_2/logs/lossfn3_metric.csv','../PIGNN_7_artery_3/logs/lossfn3_metric.csv','../PIGNN_7_artery_4/logs/lossfn3_metric.csv','../PIGNN_7_artery_5/logs/lossfn3_metric.csv','../PIGNN_7_artery_7/logs/lossfn3_metric.csv','../PIGNN_7_artery_8/logs/lossfn3_metric.csv','AR_logs','Epochs (x$10^3$)','$RRMSE$ (log)')
#plot("logs/lossfn4_metric.csv",'../PIGNN_7_artery_1/logs/lossfn4_metric.csv','../PIGNN_7_artery_2/logs/lossfn4_metric.csv','../PIGNN_7_artery_3/logs/lossfn4_metric.csv','../PIGNN_7_artery_4/logs/lossfn4_metric.csv','../PIGNN_7_artery_5/logs/lossfn4_metric.csv','../PIGNN_7_artery_7/logs/lossfn4_metric.csv','../PIGNN_7_artery_8/logs/lossfn4_metric.csv','VR_logs','Epochs (x$10^3$)','$RRMSE$')
