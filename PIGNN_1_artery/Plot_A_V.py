import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter


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


V_inlet_1,A_inlet_1,t=wave_input("../Snaps/sim_1_1.his",0,4.94,5.76)
V_outlet_1,A_outlet_1,t=wave_input("../Snaps/sim_1_1.his",1,4.94,5.76)

A=np.load('A.npy')
V=np.load('V.npy')

class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"

def plot(Time,Real,Prediction,name,Xaxis,Yaxis,legend):
	plt.rcParams["figure.autolayout"] = True

	plt.plot(Time,Real,'r-', linewidth=3,label='Ref')
	plt.plot(Time,Prediction,'b--', dashes=(5, 4), linewidth=3,label='Prediction')
	plt.xlabel(Xaxis,fontsize=20)
	plt.ylabel(Yaxis,fontsize=20)
	if legend==1:
		plt.legend(prop = { "size": 20})
	ax = plt.gca()
	#yScalarFormatter = ScalarFormatterClass(useMathText=True)
	#yScalarFormatter.set_powerlimits((0,0))
	#ax.yaxis.set_major_formatter(yScalarFormatter)
	ax.spines[['right', 'top']].set_visible(False)

	ax.tick_params(axis='both', which='major', labelsize=15)

	
	plt.savefig(name)
	plt.cla()
	plt.clf()
	

plot(t[::2]-np.min(t),A_inlet_1[::2]*1e4,A[0,::2]*1e4,'A_inlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),V_inlet_1[::2],V[0,::2],'V_inlet','Time (s)','Velocity (m/s)',1)	
plot(t[::2]-np.min(t),A_outlet_1[::2]*1e4,A[9,::2]*1e4,'A_outlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),V_outlet_1[::2],V[9,::2],'V_outlet','Time (s)','Velocity (m/s)',0)	

