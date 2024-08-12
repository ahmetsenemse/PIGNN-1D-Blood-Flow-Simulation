from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from spektral.layers import ARMAConv





def GNN(N,F,n_out,channel_number,order_number,it_number,acti,actigcn):

    x_in = Input(shape=(F,))
    a_in = Input((N,), sparse=True)
    
    gc_1 = ARMAConv(
        channel_number,
        iterations=it_number,
        order=order_number,
        share_weights=False,
        activation=acti,
        gcn_activation=actigcn,
    )([x_in, a_in])
    
    for i in range(0,4):
        gc_1 = ARMAConv(
            channel_number,
            iterations=it_number,
            order=order_number,
            share_weights=False,
            activation=acti,
            gcn_activation=actigcn,
        )([gc_1, a_in])
    gc_2 = ARMAConv(
        n_out,
        iterations=it_number,
        order=order_number,
        share_weights=False,
        activation=acti,
        gcn_activation=None,
    )([gc_1, a_in])
    
    
    model = Model(inputs=[x_in, a_in], outputs=gc_2)
    return model	



