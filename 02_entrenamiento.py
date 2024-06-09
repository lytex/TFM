# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: TFM
#     language: python
#     name: tfm
# ---

# %%
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
from LCWavelet import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate,Conv1D, Flatten,Dropout , BatchNormalization, MaxPooling1D
from tensorflow.keras.models import Model, Sequential

path = "all_data_2024-06-01/"
files = [file for file in os.listdir(path) if file.endswith(".pickle")]
lightcurves = []
for file in tqdm(files):
    global_local = LightCurveWaveletGlobalLocalCollection.from_pickle(path+file)
    try:
        getattr(global_local, "levels")
    except AttributeError:
        global_local.levels = [1, 2, 3, 4]
    lightcurves.append(global_local)

# %%
from collections import defaultdict

pliegue_par_global = defaultdict(list)
pliegue_impar_global = defaultdict(list)
pliegue_par_local = defaultdict(list)
pliegue_impar_local = defaultdict(list)

for lc in lightcurves:
    for level in lc.levels:
        pliegue_par_global[level].append(lc.pliegue_par_global.get_approximation_coefficent(level=level))
        pliegue_impar_global[level].append(lc.pliegue_impar_global.get_approximation_coefficent(level=level))
        pliegue_par_local[level].append(lc.pliegue_par_local.get_approximation_coefficent(level=level))
        pliegue_impar_local[level].append(lc.pliegue_impar_local.get_approximation_coefficent(level=level))
        

pliegue_par_global = {k: np.array(v) for k, v in pliegue_par_global.items() if k in (1 ,2)}
pliegue_par_global = {k: v.reshape(list(v.shape)+[1]) for k, v in pliegue_par_global.items()}
pliegue_impar_global = {k: np.array(v) for k, v in pliegue_impar_global.items() if k in (1, 2)}
pliegue_impar_global = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_impar_global.items()}


pliegue_par_local = {k: np.array(v) for k, v in pliegue_par_local.items() if k in (1, 2)}
pliegue_par_local = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_par_local.items()}
pliegue_impar_local = {k: np.array(v) for k, v in pliegue_impar_local.items() if k in (1, 2)}
pliegue_impar_local = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_impar_local.items()}


inputs = (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local)


  # %%
  def _build_cnn_layers(self, inputs, hparams, scope="cnn"):
    """Builds convolutional layers.

    The layers are defined by convolutional blocks with pooling between blocks
    (but not within blocks). Within a block, all layers have the same number of
    filters, which is a constant multiple of the number of filters in the
    previous block. The kernel size is fixed throughout.

    Args:
      inputs: A Tensor of shape [batch_size, length] or
        [batch_size, length, ndims].
      hparams: Object containing CNN hyperparameters.
      scope: Prefix for operation names.

    Returns:
      A Tensor of shape [batch_size, output_size], where the output size depends
      on the input size, kernel size, number of filters, number of layers,
      convolution padding type and pooling.
    """
    with tf.name_scope(scope):
      net = inputs
      if net.shape.rank == 2:
        net = tf.expand_dims(net, -1)  # [batch, length] -> [batch, length, 1]
      if net.shape.rank != 3:
        raise ValueError(
            "Expected inputs to have rank 2 or 3. Got: {}".format(inputs))
      for i in range(hparams.cnn_num_blocks):
        num_filters = int(hparams.cnn_initial_num_filters *
                          hparams.cnn_block_filter_factor**i)
        with tf.name_scope("block_{}".format(i + 1)):
          for j in range(hparams.cnn_block_size):
            conv_op = tf.keras.layers.Conv1D(
                filters=num_filters,
                kernel_size=int(hparams.cnn_kernel_size),
                padding=hparams.convolution_padding,
                activation=tf.nn.relu,
                name="conv_{}".format(j + 1))
            net = conv_op(net)

          if hparams.pool_size > 1:  # pool_size 0 or 1 denotes no pooling
            pool_op = tf.keras.layers.MaxPool1D(
                pool_size=int(hparams.pool_size),
                strides=int(hparams.pool_strides),
                name="pool")
            net = pool_op(net)

      # Flatten.
      net.shape.assert_has_rank(3)
      net_shape = net.shape.as_list()
      output_dim = net_shape[1] * net_shape[2]
      net = tf.reshape(net, [-1, output_dim], name="flatten")

    return net


# %%

# %%
# inputs
# %pdb off
def gen_model_2_levels(inputs, classes, activation = 'relu',summary=False):
    
    (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local) = inputs    

    input_shape_global = [x.shape for x in pliegue_par_global.values()]
    assert input_shape_global == [x.shape for x in pliegue_impar_global.values()]
    
    input_shape_local = [x.shape for x in pliegue_par_local.values()]
    assert input_shape_local == [x.shape for x in pliegue_impar_local.values()]


    net = defaultdict(list)
 
    for (n, data), (n_inv, data_inv) in zip(sorted(pliegue_par_global.items(), key=lambda d: d[0]), sorted(pliegue_par_global.items(), key=lambda d: -d[0])):
        block = Sequential()
        for i in range(n_inv):
            block.add( Conv1D(16*2**n_inv, 5, activation=activation, input_shape=data.shape[1:], ))
            block.add(Conv1D(16*2**n_inv, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))

        block.add(Flatten())
        net["global_par"].append(block)
        
    for (n, data), (n_inv, data_inv) in zip(sorted(pliegue_impar_global.items(), key=lambda d: d[0]), sorted(pliegue_impar_global.items(), key=lambda d: -d[0])):
        block = Sequential()
        for i in range(n_inv):
            block.add(Conv1D(16*2**n_inv, 5, activation=activation, input_shape=data.shape[1:], ))
            block.add(Conv1D(16*2**n_inv, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))

        block.add(Flatten())
        net["global_impar"].append(block)

    for (n, data), (n_inv, data_inv) in zip(sorted(pliegue_par_local.items(), key=lambda d: d[0]), sorted(pliegue_par_local.items(), key=lambda d: -d[0])):
        block = Sequential()
        for i in range(n_inv):
            block.add(Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["local_par"].append(block)
        
    for (n, data), (n_inv, data_inv) in zip(sorted(pliegue_impar_local.items(), key=lambda d: d[0]), sorted(pliegue_impar_local.items(), key=lambda d: -d[0])):
        block = Sequential()
        for i in range(n_inv):
            block.add( Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            block.add( Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["local_impar"].append(block)
                             
    model_f = concatenate([m.output for m in net["local_impar"]] + [m.output for m in net["local_par"]] + [m.output for m in net["global_impar"]] + [m.output for m in net["global_par"]], axis=-1)
    model_f = BatchNormalization(axis=-1)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(len(classes),activation='softmax')(model_f)
    
    model_f = Model([[m.input for m in net["local_impar"]], [m.input for m in net["local_par"]]  , [m.input for m in net["global_impar"]], [m.input for m in net["global_par"]]],model_f)
    # model_f = Model([[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)


    if summary:
      model_f.summary()
    return model_f
    
    # model_f = concatenate([model_p_7.output,model_i_7.output,
    #                        model_p_8.output,model_i_8.output], axis=-1)
    # model_f = BatchNormalization(axis=-1)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(1,activation='sigmoid')(model_f)
    # model_f = Model([[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)

    # tamaño nivel 7
    input_shape_1 = np.shape(pliegue_par_global[1])[1:]
    # tamaño nivel 8
    input_shape_2 = np.shape(pliegue_par_global[2])[1:]
   
    # rama par level 7
    model_p_7 = tf.keras.Sequential()
    model_p_7.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_1))
    model_p_7.add(Conv1D(16, 5, activation=activation)) 
    model_p_7.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_7.add(Conv1D(32,5, activation=activation))
    model_p_7.add(Conv1D(32,5, activation=activation))
    model_p_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_7.add(Conv1D(64,5, activation=activation))
    model_p_7.add(Conv1D(64,5, activation=activation))
    model_p_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_7.add(Conv1D(128,5, activation=activation))
    model_p_7.add(Conv1D(128,5, activation=activation))
    model_p_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_7.add(Flatten())
    
    # rama par level 8
    model_p_8 = tf.keras.Sequential()
    model_p_8.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_2))
    model_p_8.add(Conv1D(16, 5, activation=activation)) 
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_8.add(Conv1D(32,5, activation=activation))
    model_p_8.add(Conv1D(32,5, activation=activation))
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_8.add(Conv1D(64,5, activation=activation))
    model_p_8.add(Conv1D(64,5, activation=activation))
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_8.add(Flatten())
    
    # rama impar level 7
    model_i_7 = tf.keras.Sequential()
    model_i_7.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_1))
    model_i_7.add(Conv1D(16, 5, activation=activation)) 
    model_i_7.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_i_7.add(Conv1D(32,5, activation=activation))
    model_i_7.add(Conv1D(32,5, activation=activation))
    model_i_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_7.add(Conv1D(64,5, activation=activation))
    model_i_7.add(Conv1D(64,5, activation=activation))
    model_i_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_7.add(Conv1D(128,5, activation=activation))
    model_i_7.add(Conv1D(128,5, activation=activation))
    model_i_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_7.add(Flatten())
    
    # rama impar level 8
    model_i_8 = tf.keras.Sequential()
    model_i_8.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_2))
    model_i_8.add(Conv1D(16, 5, activation=activation)) 
    model_i_8.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_i_8.add(Conv1D(32,5, activation=activation))
    model_i_8.add(Conv1D(32,5, activation=activation))
    model_i_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_8.add(Conv1D(64,5, activation=activation))
    model_i_8.add(Conv1D(64,5, activation=activation))
    model_i_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_8.add(Flatten())

    # Red profunda
    model_f = concatenate([model_p_7.output,model_i_7.output,
                           model_p_8.output,model_i_8.output], axis=-1)
    
    import sys; sys.__breakpointhook__()
    model_f = BatchNormalization(axis=-1)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(1,activation='sigmoid')(model_f)
    model_f = Model([[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)
    if summary:
      model_f.summary()
    return model_f

output_classes = np.unique([lc.headers['class'] for lc in lightcurves])
model_1 = gen_model_2_levels(inputs, output_classes)
model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy','binary_crossentropy'])
# history_1 = model_1.fit(X_train_8, y_train_8, epochs=1000, batch_size=64, validation_data=(X_test_8, y_test_8))

# %%
X =  
y = np.array([lc.headers['class'] for lc in lightcurves])
# for lc in lightcurves:
#  print (lc.headers['class'])
