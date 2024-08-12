import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib import ticker

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


V_inlet_1,A_inlet_1,t=wave_input("Snaps/sim_1_1.his",0,4.94,5.76)
V_outlet_1,A_outlet_1,t=wave_input("Snaps/sim_1_1.his",1,4.94,5.76)
V_inlet_2,A_inlet_2,t=wave_input("Snaps/sim_1_2.his",0,4.94,5.76)
V_outlet_2,A_outlet_2,t=wave_input("Snaps/sim_1_2.his",1,4.94,5.76)
V_inlet_3,A_inlet_3,t=wave_input("Snaps/sim_1_3.his",0,4.94,5.76)
V_outlet_3,A_outlet_3,t=wave_input("Snaps/sim_1_3.his",1,4.94,5.76)
V_inlet_4,A_inlet_4,t=wave_input("Snaps/sim_1_6.his",0,4.94,5.76)
V_outlet_4,A_outlet_4,t=wave_input("Snaps/sim_1_6.his",1,4.94,5.76)
V_inlet_5,A_inlet_5,t=wave_input("Snaps/sim_1_7.his",0,4.94,5.76)
V_outlet_5,A_outlet_5,t=wave_input("Snaps/sim_1_7.his",1,4.94,5.76)
V_inlet_6,A_inlet_6,t=wave_input("Snaps/sim_1_4.his",0,4.94,5.76)
V_outlet_6,A_outlet_6,t=wave_input("Snaps/sim_1_4.his",1,4.94,5.76)
V_inlet_7,A_inlet_7,t=wave_input("Snaps/sim_1_5.his",0,4.94,5.76)
V_outlet_7,A_outlet_7,t=wave_input("Snaps/sim_1_5.his",1,4.94,5.76)

A1_P1=np.load('A1.npy')
V1_P1=np.load('V1.npy')
A2_P1=np.load('A2.npy')
V2_P1=np.load('V2.npy')
A3_P1=np.load('A3.npy')
V3_P1=np.load('V3.npy')
A4_P1=np.load('A4.npy')
V4_P1=np.load('V4.npy')
A5_P1=np.load('A5.npy')
V5_P1=np.load('V5.npy')
A6_P1=np.load('A6.npy')
V6_P1=np.load('V6.npy')
A7_P1=np.load('A7.npy')
V7_P1=np.load('V7.npy')

A1_P2=np.load('../PIGNN_7_artery_1/A1.npy')
V1_P2=np.load('../PIGNN_7_artery_1/V1.npy')
A2_P2=np.load('../PIGNN_7_artery_1/A2.npy')
V2_P2=np.load('../PIGNN_7_artery_1/V2.npy')
A3_P2=np.load('../PIGNN_7_artery_1/A3.npy')
V3_P2=np.load('../PIGNN_7_artery_1/V3.npy')
A4_P2=np.load('../PIGNN_7_artery_1/A4.npy')
V4_P2=np.load('../PIGNN_7_artery_1/V4.npy')
A5_P2=np.load('../PIGNN_7_artery_1/A5.npy')
V5_P2=np.load('../PIGNN_7_artery_1/V5.npy')
A6_P2=np.load('../PIGNN_7_artery_1/A6.npy')
V6_P2=np.load('../PIGNN_7_artery_1/V6.npy')
A7_P2=np.load('../PIGNN_7_artery_1/A7.npy')
V7_P2=np.load('../PIGNN_7_artery_1/V7.npy')

A1_P3=np.load('../PIGNN_7_artery_2/A1.npy')
V1_P3=np.load('../PIGNN_7_artery_2/V1.npy')
A2_P3=np.load('../PIGNN_7_artery_2/A2.npy')
V2_P3=np.load('../PIGNN_7_artery_2/V2.npy')
A3_P3=np.load('../PIGNN_7_artery_2/A3.npy')
V3_P3=np.load('../PIGNN_7_artery_2/V3.npy')
A4_P3=np.load('../PIGNN_7_artery_2/A4.npy')
V4_P3=np.load('../PIGNN_7_artery_2/V4.npy')
A5_P3=np.load('../PIGNN_7_artery_2/A5.npy')
V5_P3=np.load('../PIGNN_7_artery_2/V5.npy')
A6_P3=np.load('../PIGNN_7_artery_2/A6.npy')
V6_P3=np.load('../PIGNN_7_artery_2/V6.npy')
A7_P3=np.load('../PIGNN_7_artery_2/A7.npy')
V7_P3=np.load('../PIGNN_7_artery_2/V7.npy')

A1_P4=np.load('../PIGNN_7_artery_3/A1.npy')
V1_P4=np.load('../PIGNN_7_artery_3/V1.npy')
A2_P4=np.load('../PIGNN_7_artery_3/A2.npy')
V2_P4=np.load('../PIGNN_7_artery_3/V2.npy')
A3_P4=np.load('../PIGNN_7_artery_3/A3.npy')
V3_P4=np.load('../PIGNN_7_artery_3/V3.npy')
A4_P4=np.load('../PIGNN_7_artery_3/A4.npy')
V4_P4=np.load('../PIGNN_7_artery_3/V4.npy')
A5_P4=np.load('../PIGNN_7_artery_3/A5.npy')
V5_P4=np.load('../PIGNN_7_artery_3/V5.npy')
A6_P4=np.load('../PIGNN_7_artery_3/A6.npy')
V6_P4=np.load('../PIGNN_7_artery_3/V6.npy')
A7_P4=np.load('../PIGNN_7_artery_3/A7.npy')
V7_P4=np.load('../PIGNN_7_artery_3/V7.npy')

A1_P5=np.load('../PIGNN_7_artery_4/A1.npy')
V1_P5=np.load('../PIGNN_7_artery_4/V1.npy')
A2_P5=np.load('../PIGNN_7_artery_4/A2.npy')
V2_P5=np.load('../PIGNN_7_artery_4/V2.npy')
A3_P5=np.load('../PIGNN_7_artery_4/A3.npy')
V3_P5=np.load('../PIGNN_7_artery_4/V3.npy')
A4_P5=np.load('../PIGNN_7_artery_4/A4.npy')
V4_P5=np.load('../PIGNN_7_artery_4/V4.npy')
A5_P5=np.load('../PIGNN_7_artery_4/A5.npy')
V5_P5=np.load('../PIGNN_7_artery_4/V5.npy')
A6_P5=np.load('../PIGNN_7_artery_4/A6.npy')
V6_P5=np.load('../PIGNN_7_artery_4/V6.npy')
A7_P5=np.load('../PIGNN_7_artery_4/A7.npy')
V7_P5=np.load('../PIGNN_7_artery_4/V7.npy')

A1_P6=np.load('../PIGNN_7_artery_5/A1.npy')
V1_P6=np.load('../PIGNN_7_artery_5/V1.npy')
A2_P6=np.load('../PIGNN_7_artery_5/A2.npy')
V2_P6=np.load('../PIGNN_7_artery_5/V2.npy')
A3_P6=np.load('../PIGNN_7_artery_5/A3.npy')
V3_P6=np.load('../PIGNN_7_artery_5/V3.npy')
A4_P6=np.load('../PIGNN_7_artery_5/A4.npy')
V4_P6=np.load('../PIGNN_7_artery_5/V4.npy')
A5_P6=np.load('../PIGNN_7_artery_5/A5.npy')
V5_P6=np.load('../PIGNN_7_artery_5/V5.npy')
A6_P6=np.load('../PIGNN_7_artery_5/A6.npy')
V6_P6=np.load('../PIGNN_7_artery_5/V6.npy')
A7_P6=np.load('../PIGNN_7_artery_5/A7.npy')
V7_P6=np.load('../PIGNN_7_artery_5/V7.npy')

A1_P7=np.load('../PIGNN_7_artery_7/A1.npy')
V1_P7=np.load('../PIGNN_7_artery_7/V1.npy')
A2_P7=np.load('../PIGNN_7_artery_7/A2.npy')
V2_P7=np.load('../PIGNN_7_artery_7/V2.npy')
A3_P7=np.load('../PIGNN_7_artery_7/A3.npy')
V3_P7=np.load('../PIGNN_7_artery_7/V3.npy')
A4_P7=np.load('../PIGNN_7_artery_7/A4.npy')
V4_P7=np.load('../PIGNN_7_artery_7/V4.npy')
A5_P7=np.load('../PIGNN_7_artery_7/A5.npy')
V5_P7=np.load('../PIGNN_7_artery_7/V5.npy')
A6_P7=np.load('../PIGNN_7_artery_7/A6.npy')
V6_P7=np.load('../PIGNN_7_artery_7/V6.npy')
A7_P7=np.load('../PIGNN_7_artery_7/A7.npy')
V7_P7=np.load('../PIGNN_7_artery_7/V7.npy')

A1_P8=np.load('../PIGNN_7_artery_8/A1.npy')
V1_P8=np.load('../PIGNN_7_artery_8/V1.npy')
A2_P8=np.load('../PIGNN_7_artery_8/A2.npy')
V2_P8=np.load('../PIGNN_7_artery_8/V2.npy')
A3_P8=np.load('../PIGNN_7_artery_8/A3.npy')
V3_P8=np.load('../PIGNN_7_artery_8/V3.npy')
A4_P8=np.load('../PIGNN_7_artery_8/A4.npy')
V4_P8=np.load('../PIGNN_7_artery_8/V4.npy')
A5_P8=np.load('../PIGNN_7_artery_8/A5.npy')
V5_P8=np.load('../PIGNN_7_artery_8/V5.npy')
A6_P8=np.load('../PIGNN_7_artery_8/A6.npy')
V6_P8=np.load('../PIGNN_7_artery_8/V6.npy')
A7_P8=np.load('../PIGNN_7_artery_8/A7.npy')
V7_P8=np.load('../PIGNN_7_artery_8/V7.npy')

class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"

def plot(Time,Real,Prediction1,Prediction2,Prediction3,Prediction4,Prediction5,Prediction6,Prediction7,Prediction8,name,Xaxis,Yaxis,legend):
	plt.rcParams["figure.autolayout"] = True

	plt.plot(Time,Real,'r-', linewidth=3,label='Ref')
	plt.plot(Time,Prediction1,'b--', dashes=(5, 4), linewidth=3,label='$Scenario-1$')
	plt.plot(Time,Prediction2,'g--', dashes=(5, 4), linewidth=3,label='$Scenario-2$')
	plt.plot(Time,Prediction3,'k--', dashes=(5, 4), linewidth=3,label='$Scenario-3$')
	plt.plot(Time,Prediction4,'c--', dashes=(5, 4), linewidth=3,label='$Scenario-4$')
	plt.plot(Time,Prediction5,'m--', dashes=(5, 4), linewidth=3,label='$Scenario-5$')
	plt.plot(Time,Prediction6,'y--', dashes=(5, 4), linewidth=3,label='$Scenario-6$')
	plt.plot(Time,Prediction7,'tab:orange', dashes=(5, 4), linewidth=3,label='$Scenario-7$')
	plt.plot(Time,Prediction8,'tab:gray', dashes=(5, 4), linewidth=3,label='$Scenario-8$')
	#plt.xlabel(Xaxis,fontsize=20)
	plt.ylabel(Yaxis,fontsize=20)
	plt.tick_params(
    		axis='x',          # changes apply to the x-axis
    		which='both',      # both major and minor ticks are affected
    		bottom=False,      # ticks along the bottom edge are off
    		top=False,         # ticks along the top edge are off
    		labelbottom=False)

	
	if legend==1:
		plt.legend(prop = { "size": 20})
	ax = plt.gca()
	#yScalarFormatter = ScalarFormatterClass(useMathText=True)
	#yScalarFormatter.set_powerlimits((0,0))
	#ax.yaxis.set_major_formatter(yScalarFormatter)
	ax.spines[['right', 'top', 'bottom']].set_visible(False)

	ax.tick_params(axis='both', which='major', labelsize=15)
	#ax.legend(bbox_to_anchor=(1.1, 1.05))
	ax.yaxis.set_major_locator(plt.MaxNLocator(5))
	ax.yaxis.set_minor_locator(plt.MaxNLocator(5))
	ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
	
	plt.savefig(name)
	plt.cla()
	plt.clf()
	
"""
plot(t[::2]-np.min(t),A_outlet_1[::2]*1e4,A1_P1[9,::2]*1e4,A1_P2[9,::2]*1e4,A1_P3[9,::2]*1e4,A1_P4[9,::2]*1e4,A1_P5[9,::2]*1e4,A1_P6[9,::2]*1e4,A1_P7[9,::2]*1e4,A1_P8[9,::2]*1e4,'A1_outlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),A_inlet_2[::2]*1e4,A2_P1[0,::2]*1e4,A2_P2[0,::2]*1e4,A2_P3[0,::2]*1e4,A2_P4[0,::2]*1e4,A2_P5[0,::2]*1e4,A2_P6[0,::2]*1e4,A2_P7[0,::2]*1e4,A2_P8[0,::2]*1e4,'A2_inlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),A_inlet_3[::2]*1e4,A3_P1[0,::2]*1e4,A3_P2[0,::2]*1e4,A3_P3[0,::2]*1e4,A3_P4[0,::2]*1e4,A3_P5[0,::2]*1e4,A3_P6[0,::2]*1e4,A3_P7[0,::2]*1e4,A3_P8[0,::2]*1e4,'A3_inlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),A_inlet_4[::2]*1e4,A4_P1[0,::2]*1e4,A4_P2[0,::2]*1e4,A4_P3[0,::2]*1e4,A4_P4[0,::2]*1e4,A4_P5[0,::2]*1e4,A4_P6[0,::2]*1e4,A4_P7[0,::2]*1e4,A4_P8[0,::2]*1e4,'A4_inlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),A_inlet_5[::2]*1e4,A5_P1[0,::2]*1e4,A5_P2[0,::2]*1e4,A5_P3[0,::2]*1e4,A5_P4[0,::2]*1e4,A5_P5[0,::2]*1e4,A5_P6[0,::2]*1e4,A5_P7[0,::2]*1e4,A5_P8[0,::2]*1e4,'A5_inlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),A_inlet_6[::2]*1e4,A6_P1[0,::2]*1e4,A6_P2[0,::2]*1e4,A6_P3[0,::2]*1e4,A6_P4[0,::2]*1e4,A6_P5[0,::2]*1e4,A6_P6[0,::2]*1e4,A6_P7[0,::2]*1e4,A6_P8[0,::2]*1e4,'A6_inlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),A_inlet_7[::2]*1e4,A7_P1[0,::2]*1e4,A7_P2[0,::2]*1e4,A7_P3[0,::2]*1e4,A7_P4[0,::2]*1e4,A7_P5[0,::2]*1e4,A7_P6[0,::2]*1e4,A7_P7[0,::2]*1e4,A7_P8[0,::2]*1e4,'A7_inlet','Time (s)','Area ($cm^2$)',0)

"""
plot(t[::2]-np.min(t),V_outlet_1[::2],V1_P1[9,::2],V1_P2[9,::2],V1_P3[9,::2],V1_P4[9,::2],V1_P5[9,::2],V1_P6[9,::2],V1_P7[9,::2],V1_P8[9,::2],'V1_outlet','Time (s)','Velocity ($m/s$)',0)
plot(t[::2]-np.min(t),V_inlet_2[::2],V2_P1[0,::2],V2_P2[0,::2],V2_P3[0,::2],V2_P4[0,::2],V2_P5[0,::2],V2_P6[0,::2],V2_P7[0,::2],V2_P8[0,::2],'V2_inlet','Time (s)','Velocity ($m/s$)',0)
plot(t[::2]-np.min(t),V_inlet_3[::2],V3_P1[0,::2],V3_P2[0,::2],V3_P3[0,::2],V3_P4[0,::2],V3_P5[0,::2],V3_P6[0,::2],V3_P7[0,::2],V3_P8[0,::2],'V3_inlet','Time (s)','Velocity ($m/s$)',0)
plot(t[::2]-np.min(t),V_inlet_4[::2],V4_P1[0,::2],V4_P2[0,::2],V4_P3[0,::2],V4_P4[0,::2],V4_P5[0,::2],V4_P6[0,::2],V4_P7[0,::2],V4_P8[0,::2],'V4_inlet','Time (s)','Velocity ($m/s$)',0)
plot(t[::2]-np.min(t),V_inlet_5[::2],V5_P1[0,::2],V5_P2[0,::2],V5_P3[0,::2],V5_P4[0,::2],V5_P5[0,::2],V5_P6[0,::2],V5_P7[0,::2],V5_P8[0,::2],'V5_inlet','Time (s)','Velocity ($m/s$)',0)
plot(t[::2]-np.min(t),V_inlet_6[::2],V6_P1[0,::2],V6_P2[0,::2],V6_P3[0,::2],V6_P4[0,::2],V6_P5[0,::2],V6_P6[0,::2],V6_P7[0,::2],V6_P8[0,::2],'V6_inlet','Time (s)','Velocity ($m/s$)',0)
plot(t[::2]-np.min(t),V_inlet_7[::2],V7_P1[0,::2],V7_P2[0,::2],V7_P3[0,::2],V7_P4[0,::2],V7_P5[0,::2],V7_P6[0,::2],V7_P7[0,::2],V7_P8[0,::2],'V7_inlet','Time (s)','Velocity ($m/s$)',0)

"""

plot(t[::2]-np.min(t),A_outlet_1[::2]*1e4,A1_P1[9,::2]*1e4,A1_P2[9,::2]*1e4,A1_P3[9,::2]*1e4,A1_P4[9,::2]*1e4,A1_P5[9,::2]*1e4,A1_P6[9,::2]*1e4,A1_P7[9,::2]*1e4,A1_P8[9,::2]*1e4,'Legend1','Time (s)','Area ($cm^2$)',1)
"""

