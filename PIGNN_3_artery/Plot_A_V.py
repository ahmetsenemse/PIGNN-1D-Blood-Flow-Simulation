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
V_inlet_2,A_inlet_2,t=wave_input("../Snaps/sim_1_2.his",0,4.94,5.76)
V_outlet_2,A_outlet_2,t=wave_input("../Snaps/sim_1_2.his",1,4.94,5.76)
V_inlet_3,A_inlet_3,t=wave_input("../Snaps/sim_1_3.his",0,4.94,5.76)
V_outlet_3,A_outlet_3,t=wave_input("../Snaps/sim_1_3.his",1,4.94,5.76)

A1_PIGNN=np.load('A1.npy')
V1_PIGNN=np.load('V1.npy')
A2_PIGNN=np.load('A2.npy')
V2_PIGNN=np.load('V2.npy')
A3_PIGNN=np.load('A3.npy')
V3_PIGNN=np.load('V3.npy')


A1_PINN_i=np.load('../ClassicalPINN/inflow_A_1.npy')
V1_PINN_i=np.load('../ClassicalPINN/inflow_u_1.npy')
A2_PINN_i=np.load('../ClassicalPINN/inflow_A_2.npy')
V2_PINN_i=np.load('../ClassicalPINN/inflow_u_2.npy')
A3_PINN_i=np.load('../ClassicalPINN/inflow_A_3.npy')
V3_PINN_i=np.load('../ClassicalPINN/inflow_u_3.npy')

A1_PINN_o=np.load('../ClassicalPINN/outflow_A_1.npy')
V1_PINN_o=np.load('../ClassicalPINN/outflow_u_1.npy')
A2_PINN_o=np.load('../ClassicalPINN/outflow_A_2.npy')
V2_PINN_o=np.load('../ClassicalPINN/outflow_u_2.npy')
A3_PINN_o=np.load('../ClassicalPINN/outflow_A_3.npy')
V3_PINN_o=np.load('../ClassicalPINN/outflow_u_3.npy')


A1_PINNA_i=np.load('../ClassicalPINN_A/inflow_A_1.npy')
V1_PINNA_i=np.load('../ClassicalPINN_A/inflow_u_1.npy')
A2_PINNA_i=np.load('../ClassicalPINN_A/inflow_A_2.npy')
V2_PINNA_i=np.load('../ClassicalPINN_A/inflow_u_2.npy')
A3_PINNA_i=np.load('../ClassicalPINN_A/inflow_A_3.npy')
V3_PINNA_i=np.load('../ClassicalPINN_A/inflow_u_3.npy')

A1_PINNA_o=np.load('../ClassicalPINN_A/outflow_A_1.npy')
V1_PINNA_o=np.load('../ClassicalPINN_A/outflow_u_1.npy')
A2_PINNA_o=np.load('../ClassicalPINN_A/outflow_A_2.npy')
V2_PINNA_o=np.load('../ClassicalPINN_A/outflow_u_2.npy')
A3_PINNA_o=np.load('../ClassicalPINN_A/outflow_A_3.npy')
V3_PINNA_o=np.load('../ClassicalPINN_A/outflow_u_3.npy')


class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"

def plot(Time,Real,Prediction1,Prediction2,Prediction3,name,Xaxis,Yaxis,legend):
	plt.rcParams["figure.autolayout"] = True

	plt.plot(Time,Real,'r-', linewidth=3,label='Ref')
	plt.plot(Time,Prediction1,'b--', dashes=(5, 4), linewidth=3,label='$PIGNNs_u$')
	plt.plot(Time,Prediction2,'g--', dashes=(5, 4), linewidth=3,label='$PINNs_{u,A}$')
	plt.plot(Time,Prediction3,'k--', dashes=(5, 4), linewidth=3,label='$PINNs_{u}$')
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
	
print(V_inlet_3[::2,0].shape)

plot(t[::2]-np.min(t),A_outlet_1[::2]*1e4,A1_PIGNN[9,::2]*1e4,A1_PINN_o*1e4,A1_PINNA_o*1e4,'A1_outlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),V_outlet_1[::2],V1_PIGNN[9,::2],V1_PINN_o,V1_PINNA_o,'V1_outlet','Time (s)','Velocity (m/s)',1)

plot(t[::2]-np.min(t),A_inlet_2[::2]*1e4,A2_PIGNN[0,::2]*1e4,A2_PINN_i*1e4,A2_PINNA_i*1e4,'A2_inlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),V_inlet_2[::2],V2_PIGNN[0,::2],V2_PINN_i,V2_PINNA_i,'V2_inlet','Time (s)','Velocity (m/s)',0)

plot(t[::2]-np.min(t),A_inlet_3[::2]*1e4,A3_PIGNN[0,::2]*1e4,A3_PINN_i*1e4,A3_PINNA_i*1e4,'A3_inlet','Time (s)','Area ($cm^2$)',0)
plot(t[::2]-np.min(t),V_inlet_3[::2],(V3_PIGNN[0,::2]+V_inlet_3[::2,0])/2,V3_PINN_i,V3_PINNA_i,'V3_inlet','Time (s)','Velocity (m/s)',0)


