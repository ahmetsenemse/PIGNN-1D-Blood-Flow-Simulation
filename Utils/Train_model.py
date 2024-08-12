import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

import sys
sys.path.insert(1, '../Utils')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import datetime

from Metrics import R2_Lumen_Area_mt,R2_Velocity_mt,RRMSE_Lumen_Area_mt,RRMSE_Velocity_mt



def fit_model(loader,model,inlet_points,Test_data_A,Test_data_V,Nodes,N_t,customloss,epocs):

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=customloss(),  # To compute mean
        metrics=[R2_Lumen_Area_mt(loader,model,inlet_points,Test_data_A,Nodes,N_t),R2_Velocity_mt(loader,model,inlet_points,Test_data_V,Nodes,N_t)]#,RRMSE_Lumen_Area_mt(loader,model,inlet_points,Test_data_A,Nodes,N_t),RRMSE_Velocity_mt(loader,model,inlet_points,Test_data_V,Nodes,N_t)]
    )
    
    def scheduler(epoch, lr):
        lr=0.001
        if epoch>90000:
        	lr=0.0001
        return lr
            
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)      
    
    path_checkpoint = "training_1/cp.ckpt"
    directory_checkpoint = os.path.dirname(path_checkpoint)

    callback2 = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                     save_weights_only=True,
                                                     monitor='loss',
                                                     verbose=2,
                                                     #save_best_only=True,
                                                     save_freq=1000)
          
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)
    
    model.fit(
        loader.load(),
        steps_per_epoch=1,
        epochs=epocs,
        verbose=2,callbacks=[callback,callback2,tensorboard_callback]
    )
    return



