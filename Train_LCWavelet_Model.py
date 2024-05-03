# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: TFM
#     language: python
#     name: tfm
# ---

# %%

# %% id="GfWBD_IJKfGJ"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, concatenate,Conv1D, Flatten,Dropout , BatchNormalization, MaxPooling1D

# %% id="xwin1Ts_RROr"
dataset_path= 'waveletsG/'
df_path = 'light_curves_K_stars_filter.csv'
train_split = .80


# %% id="SYWcaeSgpx-Q"
class LightCurveWaveletFoldCollection():
  
    def __init__(self,light_curve,wavelets):
        self._light_curve = light_curve
        self._lc_w_collection = wavelets

    def get_detail_coefficent(self,level = None):
        if level != None:
            return self._lc_w_collection[level-1][1]
        return self._lc_w_collection[:][1]

    def get_approximation_coefficent(self,level = None):
        if level != None:
            return self._lc_w_collection[level-1][0]
        return self._lc_w_collection[:][0]
    
    def get_wavelets(self):
        return self._lc_w_collection

    def plot(self):
        wavelet = self._lc_w_collection
        time = self._light_curve.time.value
        data = self._light_curve.flux.value
        plt.figure(figsize=(16, 4))
        plt.plot(time,data)
        ig, axarr = plt.subplots(nrows=len(wavelet), ncols=2, figsize=(16,12))
        for i,lc_w in enumerate(wavelet):
            (data, coeff_d) = lc_w
            axarr[i, 0].plot(data, 'r')
            axarr[i, 1].plot(coeff_d, 'g')
            axarr[i, 0].set_ylabel("Level {}".format(i + 1), fontsize=14, rotation=90)
            axarr[i, 0].set_yticklabels([])
            if i == 0:
                axarr[i, 0].set_title("Approximation coefficients", fontsize=14)
                axarr[i, 1].set_title("Detail coefficients", fontsize=14)
            axarr[i, 1].set_yticklabels([])
        plt.show()

class LightCurveWaveletCollection():
    def __init__(self,id,headers,lc_par,lc_inpar):
        self.pliegue_par = lc_par
        self.pliegue_inpar = lc_inpar
        self.kepler_id = id
        self.headers = headers

    def save(self, path = ""):
        file_name = path + 'kic '+str(self.kepler_id)+'.pickle'
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    def load(path):
        with open(path, "rb") as f:
            w_loaded = pickle.load(f)
        return w_loaded

    def plot_comparative(self):
        light_curve_p = self.pliegue_par._light_curve
        light_curve_i = self.pliegue_inpar._light_curve
        w_par_Collection = self.pliegue_par
        w_inpar_Collection = self.pliegue_inpar
        wavelet_p=w_par_Collection.get_wavelets()
        wavelet_i=w_inpar_Collection.get_wavelets()
        plt.figure(figsize=(26, 8))
        plt.plot(light_curve_p.time.value,light_curve_p.flux.value,c='blue',label='par')
        plt.plot(light_curve_i.time.value,light_curve_i.flux.value,c='red',label='inpar')
        
        ig, axarr = plt.subplots(nrows=len(wavelet_p), ncols=2, figsize=(26,26))
        for i,zip_curves in enumerate(zip(wavelet_p,wavelet_i)):
            (data_p, coeff_p),(data_i, coeff_i) = zip_curves
            axarr[i, 0].plot(data_p,c='blue',label='par')
            axarr[i, 0].plot(data_i, c='red',label='inpar')
            axarr[i, 1].plot(coeff_p, c='blue',label='par')
            axarr[i, 1].plot(coeff_i, c='red',label='inpar')
            axarr[i, 0].set_ylabel("Level {}".format(i + 1), fontsize=14, rotation=90)
            axarr[i, 0].set_yticklabels([])
            if i == 0:
                axarr[i, 0].set_title("Approximation coefficients", fontsize=14)
                axarr[i, 1].set_title("Detail coefficients", fontsize=14)
            axarr[i, 1].set_yticklabels([])
        plt.show()


def fold_curve(light_curve_collection, period, epoch, sigma = 20, sigma_upper = 4):
    """
    Toma la coleccion de la curvas entregadas, las pliega y devuelve una sola con todos los datos.
    
    Parameters
    ----------
    light_curve_collection: LightCurveCollection
        coleccion de curvas de luz.
    period: float
        periodo de la orbita.
    epoch: float
        tiempo de cada transcurso.
    sigma: int
        valor de desviaciones estandas
    sigma_upper: int
        valor maximo de variacion
    Returns
    ----------
    una sola curva de luz
    """
    if not is_colab:
        lc_collection = lk.LightCurveCollection([lc.remove_outliers(sigma=20, sigma_upper=4) for lc in light_curve_collection])
    
    lc_ro = lc_collection.stitch()
    
    if is_colab:
        lc_ro = lc_ro.remove_outliers(sigma=sigma, sigma_upper=sigma_upper)
    
    lc_nonans = lc_ro.remove_nans()
    lc_fold = lc_nonans.fold(period = period,epoch_time = epoch)
    lc_odd=lc_fold[lc_fold.odd_mask]
    lc_even = lc_fold[lc_fold.even_mask]
    return lc_fold,lc_odd,lc_even

def apply_wavelet(light_curve,w_family, levels):
    time = light_curve.time.value
    data = light_curve.flux.value
    lc_wavelet = []
    for level in range(levels):
        level_w = pywt.dwt(data, w_family)
        lc_wavelet.append(level_w)
        data = level_w[0]
    return LightCurveWaveletFoldCollection(light_curve,lc_wavelet)

def load_light_curve(kepler_id,mission='Kepler'):
    kic = 'KIC '+str(kepler_id)
    lc_search = lk.search_lightcurve(kic, mission=mission)
    lc_collection = lc_search.download_all()
    return lc_collection

def cut_wavelet(lightCurve,window):
    
    time = lightCurve.time
    data = lightCurve.flux
    flux_error = lightCurve.flux_err
    index = np.argmin(np.absolute(time))
    min_w = index - int(window/2)
    max_w = index + int(window/2)+1
    time_selected = time[min_w:max_w]
    data_selected = data[min_w:max_w]
    flux_error_selected = flux_error[min_w:max_w]
    return lk.lightcurve.FoldedLightCurve(time=time_selected,flux=data_selected,flux_err=flux_error_selected)
    
def process_light_curve(kepler_id,disp,period,epoch,w_family,levels,plot = False, plot_comparative=False,save=False, path="",wavelet_window=None):
    # cargamos la curva de segun su Kepler_ID
    print("descargando curvas de luz...")
    lc_collection=load_light_curve(kepler_id)
    # aplicamos el pliege a las curvas de luz y las separamos en pares e inpares
    print('Aplicando pliegue y separando en pares e inpares....') 
    _,lc_inpar,lc_par = fold_curve(lc_collection,period,epoch)

    if not wavelet_window == None:
      print('Aplicando ventana ...')
      lc_inpar = cut_wavelet(lc_inpar,wavelet_window)
      lc_par = cut_wavelet(lc_par,wavelet_window)
    
    print('Aplicando wavelets...')
    # aplicamos wavelets a curvas par
    lc_w_par = apply_wavelet(lc_par,w_family,levels)
    # aplicamos wavelets a curvas inpar
    lc_w_inpar = apply_wavelet(lc_inpar,w_family,levels)
    headers = {
        "period": period,
        "epoch": epoch,
        "class": disp,
        "wavelet_family":w_family,
        "levels":levels,
        "window":wavelet_window
    }
    lc_wavelet_collection = LightCurveWaveletCollection(kepler_id,headers,lc_w_par,lc_w_inpar)
    if(plot):
        print('graficando wavelets obtenidas...')
        lc_w_par.plot()
        lc_w_inpar.plot()
    if(plot_comparative):
        print('graficando wavelets obtenidas...')
        lc_wavelet_collection.plot_comparative()
    if(save):
        print('guardando wavelets obtenidas...')
        lc_wavelet_collection.save(path)
    return lc_wavelet_collection

def process_dataset(df_koi,plot = False, plot_comparative = False,repeat_completed=True,completed=None):
    lc_wavelets = dict()
    lc_errors = []
    for i in range (len(df_koi)):

        koi_id,disp, period, epoch=df_koi[['kepid','koi_disposition','koi_period','koi_time0bk']].iloc[i]
        percent = i*100/(len(df_koi))
        print(f'procesando curva de luz KIC {int(koi_id)}[{disp}] [{percent:.0f}%]')
        if not repeat_completed and str(koi_id) in completed:
          print("curva de luz procesada anteriormente")
          continue
        try:
             process_light_curve(int(koi_id),disp,period,epoch,wavelet_family,level,plot= plot,plot_comparative=plot_comparative,save = save_lc, path = save_path,wavelet_window=wavelet_windows)
        except:
            lc_errors.append(koi_id)
            print(f'Error with KIC {koi_id}')
    f = open (save_path+'errors.txt','w')
    for lc_error in lc_errors:
        text = 'KIC '+str(lc_error)+'\n'
        f.write(text)
    f.close()
    return lc_errors
    


# %% id="pzak6wvKQ2VS"
def plot_results(history):
    # GRÁFICO DE LA PRECISIÓN y PERDIDA CON DATOS DE ENTRENAMIENTO
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Presición Entrenamiento')
    plt.plot(epochs, val_acc, 'b', label='Presición Validación')
    plt.title('Presición entrenamiento y test')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    plt.plot(epochs, loss, 'r',linestyle = 'dashed', label='Pérdida de Entrenamiento')
    plt.plot(epochs, val_loss, 'b',linestyle = 'dashed', label='Perdida de Validación')
    plt.title('Pérdida entrenamiento y test')
    plt.legend(loc=0)
    plt.show()
    
def load_files(path):
  completed = os.listdir(path)
  if "errors.txt" in completed:  
    completed.remove('errors.txt')
  completed_id = []
  for element in completed:
    completed_id.append(path+element)
  return completed_id

def generate_dataset_model_1(path,level=8):
  files = load_files(path)
  dataset_par =[]
  dataset_inpar= []
  labels = []
  len_points = None

  for i,file in enumerate(files):
    # output.clear()
    print(f"loading [{i*100/len(files):.0f}%] file:{file}")
    lcwC = LightCurveWaveletCollection.load(file)
    status = lcwC.headers['class']
    curva_par = lcwC.pliegue_par.get_approximation_coefficent(level=level)
    curva_inpar = lcwC.pliegue_inpar.get_approximation_coefficent(level=level)
    #print(i,len(curva_par),len(curva_inpar),['-' for x in range(int(len(curva_par)/10))])
    if len_points == None:
      len_points = len(curva_par)
    if len(curva_par)!= len_points or  len(curva_inpar)!= len_points:
      continue
    dataset_par.append(curva_par)
    #dataset_par=np.append(dataset_par,[curva_par])
    dataset_inpar.append(curva_inpar)
    #dataset_inpar=np.append(dataset_inpar,[curva_inpar])
    labels.append(0 if status == 'FALSE POSITIVE' else 1)
  
  dataset_par = np.array(dataset_par)
  dataset_inpar = np.array(dataset_inpar)
  labels = np.array(labels)
  return dataset_par,dataset_inpar,labels

def generate_dataset_model_2(path,levels=[8],show_loading = True):
  files = load_files(path)
  #print(f"file len:{len(files)}")
  labels = []
  len_points = {}
  curvas = {}

  for level in levels:
    curvas["par_"+str(level)] = []
    curvas["impar_"+str(level)] = []
    len_points[str(level)]=None

  for i,file in enumerate(files):
    skip_label = False
    if show_loading:
      print(f"loading [{i*100/len(files):.0f}%] file:{file}")
    lcwC = LightCurveWaveletCollection.load(file)
    status = lcwC.headers['class']

    for level in levels:
      curva_par = lcwC.pliegue_par.get_approximation_coefficent(level=level)
      curva_inpar = lcwC.pliegue_inpar.get_approximation_coefficent(level=level)

      if len_points[str(level)] == None:
         len_points[str(level)] = len(curva_par)
      if len(curva_par)!= len_points[str(level)] or  len(curva_inpar)!= len_points[str(level)]:
        skip_label = True
        break
      curvas["par_"+str(level)].append(curva_par)
      curvas["impar_"+str(level)].append(curva_inpar)

    if not skip_label:
      labels.append(0 if status == 'FALSE POSITIVE' else 1)
  #dataset_par = np.array(dataset_par)
  #dataset_inpar = np.array(dataset_inpar)
  for level in levels:
    curvas["par_"+str(level)] = np.array(curvas["par_"+str(level)])
    curvas["impar_"+str(level)] = np.array(curvas["impar_"+str(level)])
  labels = np.array(labels)
  #print("len curvas",len(curvas["par_"+str(levels[0])]),  " len labels", len(labels) )
  return curvas,labels


def split_dataset(dataset_p, dataset_i, labels, split=.80):
  split = int(len(labels)*split)
  print(f"before par:{np.shape(dataset_p)} impar:{np.shape(dataset_i)}, labels:{len(labels)}")
  X_p_train = dataset_p[:split]
  X_i_train = dataset_i[:split]
  y_train = labels[:split]

  X_p_test = dataset_p[split:]
  X_i_test = dataset_i[split:]
  y_test = labels[split:]

  X_p_train = np.expand_dims(X_p_train, axis=-1)
  X_i_train = np.expand_dims(X_i_train, axis=-1)
  X_p_test = np.expand_dims(X_p_test, axis=-1)
  X_i_test = np.expand_dims(X_i_test, axis=-1)
  #print(f"par:{np.shape(X_p_test)} impar:{np.shape(X_i_test)}, labels:{len(y_test)}")
  return [X_p_train, X_i_train], [X_p_test, X_i_test], y_train, y_test

def normalize_data(data):
  min = np.min(data)
  max = np.max(data)
  return (data - min)/(max-min) 

def normalize_data_2(data_p,data_i):
  min = np.min(data_p) if np.min(data_p) < np.min(data_i) else np.min(data_i)
  max = np.max(data_p) if np.max(data_p) > np.max(data_i) else np.max(data_i)
  return [(data_p - min)/(max-min) , (data_i - min)/(max-min)]

def normalize_LC(curvas_dic):
    return [ [normalize_data(curvas_dic[ list(curvas_dic.keys())[i]]),normalize_data(curvas_dic[ list(curvas_dic.keys())[i+1]]) ] for i in range(0,len(curvas_dic.keys()),2)   ] 
    # return [ normalize_data_2(curvas_dic[ list(curvas_dic.keys())[i]],curvas_dic[ list(curvas_dic.keys())[i+1]])  for i in range(0,len(curvas_dic.keys()),2)   ] 
def split_data_list(list_data,labels):
  ds_train = []
  ds_test = []
  label_train = []
  label_test = []
  first = True
  for c_par, c_impar in list_data:
    X_train, X_test, y_train, y_test = split_dataset(c_par, c_impar,labels)
    ds_train.append(X_train)
    ds_test.append(X_test)
    if first :
      label_train = y_train
      label_test = y_test
      first = False

  return ds_train,ds_test,label_train,label_test


# %% colab={"base_uri": "https://localhost:8080/"} id="xl3jfjC22rxO" outputId="659e74e8-34c0-4a2e-c5ab-6817ead11bd7"
ds_p_8,ds_i_8,label_8 = generate_dataset_model_1(dataset_path,level=8)
ds_p_8 = normalize_data(ds_p_8)
ds_i_8 = normalize_data(ds_i_8)
X_train_8, X_test_8, y_train_8, y_test_8 = split_dataset(ds_p_8, ds_i_8, label_8)

print(f"datos entrenamiento:{len(X_train_8[0])}, labels:{len(y_train_8)}")
print(f"datos validacion:{len(X_test_8[0])}, labels:{len(y_test_8)}")
print(f"input shape par:{np.shape(X_train_8[0])} , inpar:{np.shape(X_train_8[0])}")

# %% colab={"base_uri": "https://localhost:8080/"} id="Hz7o_LBD-CNB" outputId="66ac6283-1a2a-400a-b91c-5445ef023b92"
ds_p_7,ds_i_7,label_7 = generate_dataset_model_1(dataset_path,level=3)
ds_7 = normalize_data_2(ds_p_7,ds_i_7)
X_train_7, X_test_7, y_train_7, y_test_7 = split_dataset(ds_7[0], ds_7[1], label_7)

print(f"datos entrenamiento:{len(X_train_7[0])}")
print(f"datos validacion:{len(X_test_7[0])}")
print(f"input shape par:{np.shape(X_train_7[0])} , inpar:{np.shape(X_train_7[0])}")

# %% id="mZnXH6Bd5rSN"
curvas_dic_3,labels = generate_dataset_model_2(dataset_path,levels=[5,6,7])
curvas_dic_3.keys()

# %% colab={"base_uri": "https://localhost:8080/"} id="nvGj6IGYzePi" outputId="76a6c2e4-193e-45bd-e38b-c100feb1aad0"
ds_3_levels = normalize_LC(curvas_dic_3)


# %%
[v.shape for k, v in curvas_dic_3.items()]

# %% colab={"base_uri": "https://localhost:8080/"} id="PyEnskFn2FYE" outputId="db550fc6-2d5d-4db7-ab4d-8c3647e963d8"
ds_3_train,ds_3_test,label_3_train,label_3_test = split_data_list(ds_3_levels,labels)
np.shape(ds_3_train[2])
print(f"labels total:{len(labels)}")
print(f"datos entrenamiento:{len(ds_3_train[0][0])}, labels:{len(label_3_train)}")
print(f"datos validacion:{len(ds_3_test[0][0])}, labels:{len(label_3_test)}")

# %% id="p89n08-tQ2V_"
curvas_dic_4,labels_4 = generate_dataset_model_2(dataset_path,levels=[5,6,7,8])
curvas_dic_4.keys()

# %% colab={"base_uri": "https://localhost:8080/"} id="xf5rWf-8RR3B" outputId="0e36c01c-37b6-42f9-fc4f-e52666902fc7"
curvas_dic_4
# Está dando error normalize_LC
ds_4_levels = normalize_LC(curvas_dic_4)
# print(np.shape(ds_4_levels))

# %% colab={"base_uri": "https://localhost:8080/"} id="HjW3eP5tSN4l" outputId="bbbc3f95-0a46-405e-98ab-0651d668b05f"
ds_4_train,ds_4_test,label_4_train,label_4_test = split_data_list(ds_4_levels,labels_4)
np.shape(ds_4_train[2])


# %% [markdown] id="4QdUCI1qmkzS"
# # Construnccion del modelo 1
#     [Conv1D(16,5)]    [Conv1D(16,5)]
#     [Conv1D(16,5)]    [Conv1D(16,5)]
#     [MaxPool(3,2)]    [MaxPool(3,2)]
#     [Conv1D(32,5)]    [Conv1D(32,5)]
#     [Conv1D(32,5)]    [Conv1D(32,5)]
#     [MaxPool(3,2)]    [MaxPool(3,2)]
#       [Flatten()]      [Flatten()]
#                 [Concat]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(1sigmoid)]
#
#
#
#

# %% id="3QQJ5W2d4gfZ"
def gen_model_1_level(ds,activation = 'relu'): 
  input_shape = np.shape(ds)[2:]
  print(input_shape)
  model_p = tf.keras.Sequential()
  model_p.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=input_shape))
  model_p.add(Conv1D(filters=16, kernel_size=5, activation='relu')) 
  model_p.add(MaxPooling1D(pool_size=3, strides=1)) 
  model_p.add(Conv1D(32,5, activation='relu'))
  model_p.add(Conv1D(32,5, activation='relu'))
  model_p.add(MaxPooling1D(pool_size=3, strides=1))
  model_p.add(Flatten())

  model_i = tf.keras.Sequential()
  model_i.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=input_shape))
  model_i.add(Conv1D(filters=16, kernel_size=5, activation='relu')) 
  model_i.add(MaxPooling1D(pool_size=3, strides=1)) 
  model_i.add(Conv1D(32,5, activation='relu'))
  model_i.add(Conv1D(32,5, activation='relu'))
  model_i.add(MaxPooling1D(pool_size=3, strides=1))
  model_i.add(Flatten())

  model_f = concatenate([model_p.output,model_i.output], axis=-1)
  model_f = Dense(512,activation='relu')(model_f)
  model_f = Dense(512,activation='relu')(model_f)
  model_f = Dense(512,activation='relu')(model_f)
  model_f = Dense(512,activation='relu')(model_f)
  model_f = Dense(1,activation='sigmoid')(model_f)
  model_f = Model([model_p.input,model_i.input],model_f)
  return model_f


# %% colab={"base_uri": "https://localhost:8080/", "height": 563} id="s-vaJmZIe9dM" outputId="bf223032-ebbc-4286-b623-64cf808cb4cb"
# plot_results(history_1)
ds_7[0].shape


# %% id="NG5-j8MbAJRe"
# model_f =  gen_model_1_level(X_train_7)
# test_loss, test_acc = model_f.evaluate(X_test_7,  y_test_7)

# print("------------------------------")
# print("Métricas de validación")
# print("------------------------------")
# print("Pérdida: %.2f" % (test_loss*100))
# print( "Precisión: %.2f" % (test_acc*100))

# predictions = model_f.predict(X_test_7)

# for pred,real in zip(predictions,y_test_7):
#   print(pred,real) 

# %% [markdown] id="siPYumaAruqp"
# # Construnccion del modelo 2
#     [Conv1D(16,5)]    [Conv1D(16,5)]
#     [Conv1D(16,5)]    [Conv1D(16,5)]
#     [MaxPool(3,2)]    [MaxPool(3,2)]
#     [Conv1D(32,5)]    [Conv1D(32,5)]
#     [Conv1D(32,5)]    [Conv1D(32,5)]
#     [MaxPool(3,2)]    [MaxPool(3,2)]
#     [Conv1D(64,5)]    [Conv1D(64,5)]
#     [Conv1D(64,5)]    [Conv1D(64,5)]
#     [MaxPool(3,2)]    [MaxPool(3,2)]
#       [Flatten()]      [Flatten()]
#                 [Concat]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(1sigmoid)]
#
#
#
#

# %% id="Obg9qPdIrr4-"
def gen_model_1_level_2(ds,activation = 'relu'): 
  input_shape = np.shape(ds)[2:]
  model_p = tf.keras.Sequential()
  model_p.add(Conv1D(filters=32, kernel_size=5, input_shape=input_shape))
  model_p.add(Conv1D(filters=32, kernel_size=5, )) 
  model_p.add(MaxPooling1D(pool_size=3, strides=1)) 
  model_p.add(Conv1D(64,5))
  model_p.add(Conv1D(64,5))
  model_p.add(MaxPooling1D(pool_size=3, strides=1))
  model_p.add(Conv1D(128,5))
  model_p.add(Conv1D(128,5))
  model_p.add(MaxPooling1D(pool_size=3, strides=1))
  model_p.add(Flatten())

  model_i = tf.keras.Sequential()
  model_i.add(Conv1D(filters=32, kernel_size=5,  input_shape=input_shape))
  model_i.add(Conv1D(filters=32, kernel_size=5,)) 
  model_i.add(MaxPooling1D(pool_size=3, strides=1)) 
  model_i.add(Conv1D(64,5))
  model_i.add(Conv1D(64,5))
  model_i.add(MaxPooling1D(pool_size=3, strides=1))
  model_i.add(Conv1D(128,5))
  model_i.add(Conv1D(128,5))
  model_i.add(MaxPooling1D(pool_size=3, strides=1))
  model_i.add(Flatten())

  model_f = concatenate([model_p.output,model_i.output], axis=-1)
  model_f = BatchNormalization(axis=-1)(model_f)
  model_f = Dense(256,activation='relu')(model_f)
  model_f = Dense(256,activation='relu')(model_f)
  model_f = Dropout(.2)(model_f)
  model_f = Dense(256,activation='relu')(model_f)
  model_f = Dense(256,activation='relu')(model_f)
  model_f = Dropout(.2)(model_f)
  model_f = Dense(256,activation='relu')(model_f)
  model_f = Dense(1,activation='sigmoid')(model_f)
  model_f_2 = Model(inputs=[model_p.input,model_i.input],outputs=model_f)
  return model_f_2


# %% [markdown] id="g_dLcwI7tiwV"
# # Construnccion del modelo 3
#
#     [Conv1D(16,5)]                                        [Conv1D(16,5)]
#     [Conv1D(16,5)]                                        [Conv1D(16,5)]
#     [MaxPool(3,1)]                                        [MaxPool(3,1)]
#     [Conv1D(32,5)]    [Conv1D(16,5)]    [Conv1D(16,5)]    [Conv1D(32,5)]
#     [Conv1D(32,5)]    [Conv1D(16,5)]    [Conv1D(16,5)]    [Conv1D(32,5)]
#     [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]
#     [Conv1D(64,5)]    [Conv1D(32,5)]    [Conv1D(32,5)]    [Conv1D(64,5)]
#     [Conv1D(64,5)]    [Conv1D(32,5)]    [Conv1D(32,5)]    [Conv1D(64,5)]
#     [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]
#     [Conv1D(128,5)]   [Conv1D(64,5)]    [Conv1D(64,5)]    [Conv1D(128,5)]
#     [Conv1D(128,5)]   [Conv1D(64,5)]    [Conv1D(64,5)]    [Conv1D(128,5)]
#     [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]
#       [Flatten()]      [Flatten()]        [Flatten()]      [Flatten()]
#                                   [Concat]
#                               [Dense(512relu)]
#                               [Dense(512relu)]
#                               [Dense(512relu)]
#                               [Dense(512relu)]
#                               [Dense(1sigmoid)]

# %% id="KDtSmJr_vLSJ"
def gen_model_2_levels(ds,activation = 'relu',summary=False):
    # tamaño nivel 7
    input_shape_1 = np.shape(ds[0])[2:]
    # tamaño nivel 8
    input_shape_2 = np.shape(ds[1])[2:]
   
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 223} id="53s0EX4Ex4S_" outputId="b5f91e89-6ac8-4cde-be1f-e014faef9800"
# model_3 =  gen_model_3()
# model_3.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy','binary_crossentropy'])
# history_2 = model_3.fit([X_train_7,X_train_8], y_train_7, epochs=1000, batch_size=64,validation_split=0.15,shuffle=True)
# plot_results(history_2)

# %% [markdown] id="CMs4doFD4Tcr"
# # Construnccion del modelo 4
#
# 		[Conv1D(16,5)]				                                          					[Conv1D(16,5)]
# 		[Conv1D(16,5)]					                                        				[Conv1D(16,5)]
# 		[MaxPool(3,1)]					                                        				[MaxPool(3,1)]
# 		[Conv1D(32,5)]	[Conv1D(16,5)]                                        [Conv1D(16,5)]	[Conv1D(32,5)]
# 		[Conv1D(32,5)]	[Conv1D(16,5)]                                        [Conv1D(16,5)]	[Conv1D(32,5)]
# 		[MaxPool(3,5)]	[MaxPool(3,1)]                                        [MaxPool(3,1)]	[MaxPool(3,1)]
# 		[Conv1D(64,5)]	[Conv1D(32,5)]    [Conv1D(16,5)]    [Conv1D(16,5)]    [Conv1D(32,5)]	[Conv1D(64,5)]
# 		[Conv1D(64,5)]	[Conv1D(32,5)]    [Conv1D(16,5)]    [Conv1D(16,5)]    [Conv1D(32,5)]	[Conv1D(64,5)]
# 		[MaxPool(3,5)]	[MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]	[MaxPool(3,1)]
# 		[Conv1D(128,5)]	[Conv1D(64,5)]    [Conv1D(32,5)]    [Conv1D(32,5)]    [Conv1D(64,5)]	[Conv1D(128,5)]
# 		[Conv1D(12u,5)]	[Conv1D(64,5)]    [Conv1D(32,5)]    [Conv1D(32,5)]    [Conv1D(64,5)]	[Conv1D(128,5)]
# 		[MaxPool(3,5)]	[MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]	[MaxPool(3,1)]
# 		[Conv1D(256,5)]	[Conv1D(128,5)]   [Conv1D(64,5)]    [Conv1D(64,5)]    [Conv1D(128,5)]	[Conv1D(256,5)]
# 		[Conv1D(256,5)]	[Conv1D(128,5)]   [Conv1D(64,5)]    [Conv1D(64,5)]    [Conv1D(128,5)]	[Conv1D(256,5)]
# 		[MaxPool(3,1)]	[MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]	[MaxPool(16,5)]
# 		 [Flatten()]	  [Flatten()]      [Flatten()]        [Flatten()]      [Flatten()]		[Flatten()]
#                                                         [Concat]
#                                                       [Dense(512relu)]
#                                                       [Dense(512relu)]
#                                                       [Dense(512relu)]
#                                                       [Dense(512relu)]
#                                                       [Dense(1sigmoid)]

# %% id="rNrOjswE65dz"
def gen_model_3_levels(ds,activation = 'relu', print_summary = False):
    # tamaño nivel 3
    input_shape_3 = np.shape(ds[0])[2:]
    # tamaño nivel 7
    input_shape_7 = np.shape(ds[1])[2:]
    # tamaño nivel 8
    input_shape_8 = np.shape(ds[2])[2:]
   
     # rama par level 3
    model_p_3 = tf.keras.Sequential()
    model_p_3.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_3))
    model_p_3.add(Conv1D(16, 5, activation=activation)) 
    model_p_3.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_3.add(Conv1D(32,5, activation=activation))
    model_p_3.add(Conv1D(32,5, activation=activation))
    model_p_3.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_3.add(Conv1D(64,5, activation=activation))
    model_p_3.add(Conv1D(64,5, activation=activation))
    model_p_3.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_3.add(Conv1D(128,5, activation=activation))
    model_p_3.add(Conv1D(128,5, activation=activation))
    model_p_3.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_3.add(Conv1D(128,5, activation=activation))
    model_p_3.add(Conv1D(128,5, activation=activation))
    model_p_3.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_3.add(Flatten())

    # rama par level 7
    model_p_7 = tf.keras.Sequential()
    model_p_7.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_7))
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
    model_p_8.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_8))
    model_p_8.add(Conv1D(16, 5, activation=activation)) 
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_8.add(Conv1D(32,5, activation=activation))
    model_p_8.add(Conv1D(32,5, activation=activation))
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_8.add(Conv1D(64,5, activation=activation))
    model_p_8.add(Conv1D(64,5, activation=activation))
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_8.add(Flatten())
    
     # rama impar level 3
    model_i_3 = tf.keras.Sequential()
    model_i_3.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_3))
    model_i_3.add(Conv1D(16, 5, activation=activation)) 
    model_i_3.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_i_3.add(Conv1D(32,5, activation=activation))
    model_i_3.add(Conv1D(32,5, activation=activation))
    model_i_3.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_3.add(Conv1D(64,5, activation=activation))
    model_i_3.add(Conv1D(64,5, activation=activation))
    model_i_3.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_3.add(Conv1D(128,5, activation=activation))
    model_i_3.add(Conv1D(128,5, activation=activation))
    model_i_3.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_3.add(Conv1D(128,5, activation=activation))
    model_i_3.add(Conv1D(128,5, activation=activation))
    model_i_3.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_3.add(Flatten())

    # rama impar level 7
    model_i_7 = tf.keras.Sequential()
    model_i_7.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_7))
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
    model_i_8.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_8))
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
    model_f = concatenate([model_p_3.output,model_i_3.output,
                           model_p_7.output,model_i_7.output,
                           model_p_8.output,model_i_8.output], axis=-1)
    
    model_f = BatchNormalization(axis=-1)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dropout(0.2)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dropout(0.2)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(1,activation='sigmoid')(model_f)
    model_f = Model([[model_p_3.input,model_i_3.input],[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)
    if print_summary:
      model_f.summary()
    return model_f


# %% id="PK2nPmCM-L6b"
model_4 =  gen_model_3_levels(ds_3_train,activation = tf.keras.layers.LeakyReLU())
model_4.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy','binary_crossentropy'])
history_4 = model_4.fit(ds_3_train, label_3_train, epochs=1000, batch_size=64,validation_split=0.15,shuffle=True)
plot_results(history_4)


# %% [markdown] id="jn2umUMDODBd"
# # Construnccion modelo de 4 niveles
#
# 		[Conv1D(16,5)]				                                          					[Conv1D(16,5)]
# 		[Conv1D(16,5)]					                                        				[Conv1D(16,5)]
# 		[MaxPool(3,1)]					                                        				[MaxPool(3,1)]
# 		[Conv1D(32,5)]	[Conv1D(16,5)]                                        [Conv1D(16,5)]	[Conv1D(32,5)]
# 		[Conv1D(32,5)]	[Conv1D(16,5)]                                        [Conv1D(16,5)]	[Conv1D(32,5)]
# 		[MaxPool(3,5)]	[MaxPool(3,1)]                                        [MaxPool(3,1)]	[MaxPool(3,1)]
# 		[Conv1D(64,5)]	[Conv1D(32,5)]    [Conv1D(16,5)]    [Conv1D(16,5)]    [Conv1D(32,5)]	[Conv1D(64,5)]
# 		[Conv1D(64,5)]	[Conv1D(32,5)]    [Conv1D(16,5)]    [Conv1D(16,5)]    [Conv1D(32,5)]	[Conv1D(64,5)]
# 		[MaxPool(3,5)]	[MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]	[MaxPool(3,1)]
# 		[Conv1D(128,5)]	[Conv1D(64,5)]    [Conv1D(32,5)]    [Conv1D(32,5)]    [Conv1D(64,5)]	[Conv1D(128,5)]
# 		[Conv1D(12u,5)]	[Conv1D(64,5)]    [Conv1D(32,5)]    [Conv1D(32,5)]    [Conv1D(64,5)]	[Conv1D(128,5)]
# 		[MaxPool(3,5)]	[MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]	[MaxPool(3,1)]
# 		[Conv1D(256,5)]	[Conv1D(128,5)]   [Conv1D(64,5)]    [Conv1D(64,5)]    [Conv1D(128,5)]	[Conv1D(256,5)]
# 		[Conv1D(256,5)]	[Conv1D(128,5)]   [Conv1D(64,5)]    [Conv1D(64,5)]    [Conv1D(128,5)]	[Conv1D(256,5)]
# 		[MaxPool(3,1)]	[MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]    [MaxPool(3,1)]	[MaxPool(16,5)]
# 		 [Flatten()]	  [Flatten()]      [Flatten()]        [Flatten()]      [Flatten()]		[Flatten()]
#                                                         [Concat]
#                                                       [Dense(512relu)]
#                                                       [Dense(512relu)]
#                                                       [Dense(512relu)]
#                                                       [Dense(512relu)]
#                                                       [Dense(1sigmoid)]

# %% id="2_hld4M3ODXu"
def gen_model_4_levels(ds_train,activation = 'relu'):
    # tamaño nivel 5
    input_shape_5 = np.shape(ds_train[0])[2:]
    # tamaño nivel 6
    input_shape_6 = np.shape(ds_train[1])[2:]
    # tamaño nivel 7
    input_shape_7 = np.shape(ds_train[2])[2:]
    # tamaño nivel 8
    input_shape_8 = np.shape(ds_train[3])[2:]

     # rama par level 5
    model_p_5 = tf.keras.Sequential()
    model_p_5.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_5))
    model_p_5.add(Conv1D(16, 5, activation=activation)) 
    model_p_5.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_5.add(Conv1D(32,5, activation=activation))
    model_p_5.add(Conv1D(32,5, activation=activation))
    model_p_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_5.add(Conv1D(64,5, activation=activation))
    model_p_5.add(Conv1D(64,5, activation=activation))
    model_p_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_5.add(Conv1D(128,5, activation=activation))
    model_p_5.add(Conv1D(128,5, activation=activation))
    model_p_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_5.add(Conv1D(128,5, activation=activation))
    model_p_5.add(Conv1D(128,5, activation=activation))
    model_p_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_5.add(Conv1D(256,5, activation=activation))
    model_p_5.add(Conv1D(256,5, activation=activation))
    model_p_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_5.add(Conv1D(256,5, activation=activation))
    model_p_5.add(Conv1D(256,5, activation=activation))
    model_p_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_5.add(Flatten())

    # rama par level 6
    model_p_6 = tf.keras.Sequential()
    model_p_6.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_6))
    model_p_6.add(Conv1D(16, 5, activation=activation)) 
    model_p_6.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_6.add(Conv1D(32,5, activation=activation))
    model_p_6.add(Conv1D(32,5, activation=activation))
    model_p_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_6.add(Conv1D(64,5, activation=activation))
    model_p_6.add(Conv1D(64,5, activation=activation))
    model_p_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_6.add(Conv1D(128,5, activation=activation))
    model_p_6.add(Conv1D(128,5, activation=activation))
    model_p_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_6.add(Conv1D(128,5, activation=activation))
    model_p_6.add(Conv1D(128,5, activation=activation))
    model_p_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_6.add(Conv1D(256,5, activation=activation))
    model_p_6.add(Conv1D(256,5, activation=activation))
    model_p_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_6.add(Flatten())

    # rama par level 7
    model_p_7 = tf.keras.Sequential()
    model_p_7.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_7))
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
    model_p_8.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_8))
    model_p_8.add(Conv1D(16, 5, activation=activation)) 
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_8.add(Conv1D(32,5, activation=activation))
    model_p_8.add(Conv1D(32,5, activation=activation))
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_8.add(Conv1D(64,5, activation=activation))
    model_p_8.add(Conv1D(64,5, activation=activation))
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_8.add(Flatten())
    
     # rama impar level 5
    model_i_5 = tf.keras.Sequential()
    model_i_5.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_5))
    model_i_5.add(Conv1D(16, 5, activation=activation)) 
    model_i_5.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_i_5.add(Conv1D(32,5, activation=activation))
    model_i_5.add(Conv1D(32,5, activation=activation))
    model_i_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_5.add(Conv1D(64,5, activation=activation))
    model_i_5.add(Conv1D(64,5, activation=activation))
    model_i_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_5.add(Conv1D(128,5, activation=activation))
    model_i_5.add(Conv1D(128,5, activation=activation))
    model_i_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_5.add(Conv1D(128,5, activation=activation))
    model_i_5.add(Conv1D(128,5, activation=activation))
    model_i_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_5.add(Conv1D(256,5, activation=activation))
    model_i_5.add(Conv1D(256,5, activation=activation))
    model_i_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_5.add(Conv1D(256,5, activation=activation))
    model_i_5.add(Conv1D(256,5, activation=activation))
    model_i_5.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_5.add(Flatten())

    # rama impar level 6
    model_i_6 = tf.keras.Sequential()
    model_i_6.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_6))
    model_i_6.add(Conv1D(16, 5, activation=activation)) 
    model_i_6.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_i_6.add(Conv1D(32,5, activation=activation))
    model_i_6.add(Conv1D(32,5, activation=activation))
    model_i_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_6.add(Conv1D(64,5, activation=activation))
    model_i_6.add(Conv1D(64,5, activation=activation))
    model_i_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_6.add(Conv1D(128,5, activation=activation))
    model_i_6.add(Conv1D(128,5, activation=activation))
    model_i_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_6.add(Conv1D(128,5, activation=activation))
    model_i_6.add(Conv1D(128,5, activation=activation))
    model_i_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_6.add(Conv1D(256,5, activation=activation))
    model_i_6.add(Conv1D(256,5, activation=activation))
    model_i_6.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_6.add(Flatten())

    # rama impar level 7
    model_i_7 = tf.keras.Sequential()
    model_i_7.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_7))
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
    model_i_8.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_8))
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
    model_f = concatenate([model_p_5.output,model_i_5.output,
                           model_p_6.output,model_i_6.output,
                           model_p_7.output,model_i_7.output,
                           model_p_8.output,model_i_8.output], axis=-1)
    model_f = Dense(1012,activation=activation)(model_f)
    model_f = Dense(750,activation=activation)(model_f)
    model_f = Dropout(0.2)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dropout(0.2)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(1,activation='sigmoid')(model_f)
    model_f = Model([[model_p_5.input,model_i_5.input],[model_p_6.input,model_i_6.input],[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)
    model_f.summary()
    return model_f


# %% id="JethGx4LOFeD"
model_5 =  gen_model_4_levels(ds_4_train, activation = tf.keras.layers.LeakyReLU())
model_5.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy','binary_crossentropy'])
history_5 = model_5.fit(ds_4_train, label_4_train, epochs=1000, batch_size=64,validation_split=0.15,shuffle=True)
plot_results(history_5)

# %% id="DBYOWvTCfLun"

# %% [markdown] id="8A-xK7bDeHIV"
# # Modelo de prueba de conv1D

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="mr-4L5ywtxq3" outputId="3fe7bd3f-d741-4e7d-f757-816f9cb9fa36"
model = tf.keras.Sequential()
model.add(Conv1D(filters=16, kernel_size=5, activation='sigmoid', input_shape=(59, 1)))
model.add(Conv1D(filters=16, kernel_size=5, activation='sigmoid')) 
model.add(MaxPooling1D(pool_size=3, strides=2)) # change to 2 or add `padding="same"` to the conv layers
model.add(Conv1D(32,5, activation='sigmoid'))
model.add(Conv1D(32,5, activation='sigmoid'))
model.add(MaxPooling1D(pool_size=3, strides=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist_test = model.fit(X_train_8[0], y_train_8, epochs=50, batch_size=32,validation_split=0.3,shuffle=True)

# %% id="hhMbJnqndEG2"
plot_results(hist_test)

# %% id="vAEoL5Vscyvz"
_, accuracy = model.evaluate(X_test[0], y_test)
print('Accuracy: %.2f' % (accuracy*100))

# %% [markdown] id="di4M9N1qa9p2"
# # Pruebas
#

# %% [markdown] id="twRMz1xvkoDl"
# ## 1 nivel

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="tyudQ5b8rac0" outputId="854f5306-d526-4963-eb64-de94630698a9"
evaluate_model_1_level(gen_model_1_level_2,dataset_path,3,verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="kn-k-vbcttI9" outputId="32bf0f53-ef7e-471f-e363-73520ccb1d7b"
evaluate_model_1_level(gen_model_1_level_2,dataset_path,4,verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="ImBI2j2x7sr3" outputId="ab60b848-8420-40af-a7b4-1a82c2bac095"
evaluate_model_1_level(gen_model_1_level_2,dataset_path,5,verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="xbb0qyKcAM4Q" outputId="c13894d0-1fb5-4bcf-9b76-4aa96aed33da"
evaluate_model_1_level(gen_model_1_level_2,dataset_path,6,verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="kxShttO3AYSL" outputId="0b5dc05e-086b-45ae-9a7d-ecc6f13df8e5"
evaluate_model_1_level(gen_model_1_level_2,dataset_path,7,verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="hvlcqjhDDe7M" outputId="f0dec2e1-4a20-45cc-cc63-8d044c649b3c"
evaluate_model_1_level(gen_model_1_level_2,dataset_path,8,verbose=2)

# %% [markdown] id="XnqI_BVoG_aV"
# ## 2 niveles

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1OSauS2eHFng" outputId="2bfa12af-f37b-4dbf-d1de-4e50ec7490eb"
evaluate_model(gen_model_2_levels,generate_dataset_model_2(dataset_path,levels=[3,4],show_loading=False),verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Q5f1CSASHFhv" outputId="aa4fe3f9-9af4-4042-b946-56c6d16d2bda"
evaluate_model(gen_model_2_levels,generate_dataset_model_2(dataset_path,levels=[4,5],show_loading=False),verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="qIhRxWgFHFSL" outputId="6dcf0c9e-63b8-498a-adc9-3d392e10acbc"
evaluate_model(gen_model_2_levels,generate_dataset_model_2(dataset_path,levels=[5,6],show_loading=False),verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 169} id="V5aZHs2IHLja" outputId="14112159-9818-4c0f-fa2d-f299b030e7a2"
evaluate_model(gen_model_2_levels,generate_dataset_model_2(dataset_path,levels=[6,7],show_loading=False),verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="bORjsYFTHM5R" outputId="be7189e9-45a0-460a-e938-af21213f46f0"
evaluate_model(gen_model_2_levels,generate_dataset_model_2(dataset_path,levels=[7,8],show_loading=False),verbose=2)


# %% [markdown] id="ikhrOF3sbeyD"
# ## 3 niveles

# %% id="_a1jtXLjgCT-"
def evaluate_model(model,dataset,verbose = 0,epochs=1000):
  # Modelado Dataset
  print('normalizando datos...')
  ds_levels = normalize_LC(dataset[0])
  print('dividiendo datos en entrenamioento y test...')
  ds_train,ds_test,label_train,label_test = split_data_list(ds_levels,dataset[1])
  # entrenamiento
  print('generando modelo...')
  model_g = model(ds_train,activation = tf.keras.layers.LeakyReLU())
  print('compilando modelo....')
  model_g.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy','binary_crossentropy'])
  print('entrenando....')
  #early_stopping_acc = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.0005, patience=30, mode='max', verbose = 1)#EarlyStopping(monitor='accuracy', patience=15, min_delta=0.005, mode='max')
  early_stopping_loss = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0005, patience=15, mode='min', verbose = 1)
  history = model_g.fit(ds_train, label_train, epochs=epochs, batch_size=64,validation_split=0.20,shuffle=True,verbose=verbose, callbacks=[early_stopping_loss])
  print('obteniendo resultados..')
  plot_results(history)
  result = model_g.evaluate(ds_test, label_test)
  print(result)
  del model_g, history, result, ds_train,ds_test,label_train, label_test, ds_levels
  
def evaluate_model_1_level(model,dataset,level,verbose = 0):
  print('Cargando dataset...')
  ds_p,ds_i,label = generate_dataset_model_1(dataset_path,level)
  print('normalizando datos...')
  ds = normalize_data_2(ds_p,ds_i)
  print('dividiendo dataset...')
  X_train, X_test, y_train, y_test = split_dataset(ds[0], ds[1], label)
  print('generando modelo....')
  model_g = model(X_train,activation = tf.keras.layers.LeakyReLU())
  model_g.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
  print('entrenando...')
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=30, min_delta=0.005, mode='max')

  history_2 = model_g.fit(X_train, y_train, epochs=1000, batch_size=64,validation_split=0.15,shuffle=True,verbose = verbose, callbacks=[early_stopping])
  print('Resultados...')
  plot_results(history_2)
  _, accuracy = model_g.evaluate(X_test, y_test)
  print('Accuracy: %.2f' % (accuracy*100))

  y_prediction =[0 if x <= 0.5 else 1 for x in model_g.predict(X_test) ]
  result = confusion_matrix(y_test, y_prediction)
  disp = ConfusionMatrixDisplay(confusion_matrix=result)
  disp.plot()
  plt.show()
  del  X_train, X_test, y_train, y_test, ds_p, ds_i, ds, label, model_g, history_2, accuracy
  


# %% colab={"base_uri": "https://localhost:8080/"} id="_0tQzptJ-BC6" outputId="71824cdd-4405-4952-a8ee-1378b7ce7235"
evaluate_model(gen_model_3_levels,generate_dataset_model_2(dataset_path,levels=[3,4,5],show_loading=False),verbose=2)

# %% id="X-zd8oRkV6qP"
evaluate_model(gen_model_3_levels,generate_dataset_model_2(dataset_path,levels=[4,5,6],show_loading=False),verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1YhLxoihlP1C" outputId="1be72d79-6e50-468d-b577-88374dcb0aab"
evaluate_model(gen_model_3_levels,generate_dataset_model_2(dataset_path,levels=[5,6,7],show_loading=False),verbose=2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="OnsF0JLfgvj4" outputId="1196c870-8ce5-4914-b102-c41a957d5786"
evaluate_model(gen_model_3_levels,generate_dataset_model_2(dataset_path,levels=[6,7,8],show_loading=False),verbose=2)

# %% [markdown] id="GWE1yEDFktnI"
# ## 4 niveles
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="uHB05l1jkwf6" outputId="44003222-da5a-4362-8d9f-1789aaa44694"
evaluate_model(gen_model_4_levels,generate_dataset_model_2(dataset_path,levels=[5,6,7,8],show_loading=False),verbose=2)

# %% colab={"base_uri": "https://localhost:8080/"} id="WsD2v26m6_9G" outputId="7754e11d-c6e6-4041-8f16-8e26a704a2c3"
evaluate_model(gen_model_4_levels,generate_dataset_model_2(dataset_path,levels=[4,5,6,7],show_loading=False),verbose=2)

# %% [markdown] id="p0faydHcGOjB"
# ## Otros
