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

# %% id="3f23a54d"
import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import os
# import mathplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

# %% colab={"base_uri": "https://localhost:8080/"} id="4ead2c88" outputId="1451f1be-940f-471d-d084-e77c20c8e68b"
print(lk.__version__)
is_colab = False

# %% colab={"base_uri": "https://localhost:8080/"} id="nrOANxb9Ft2A" outputId="7a7d199a-cfdd-469e-eb32-7df5016e317a"
print(pywt.__version__)

# %% colab={"base_uri": "https://localhost:8080/"} id="8TdxE-Set5Yh" outputId="f2513b28-629e-4839-cd1c-b6cfdb875c73"
# from google.colab import drive
# drive.mount('/content/drive')

# %% id="d4faf824"
df = pd.read_csv('light_curves_K_stars_filter.csv')

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="MV1a1p4V4pad" outputId="3b2b4f97-6837-4b75-dae7-3c498713a55f"
df

# %% id="6990a68c"
# parametros para la ejecucion
wavelet_family='sym5'
level = 9
save_lc = True
save_path = 'waveletsG/'
kep_id = 10583180
kep_id_2 =9021075
wavelet_windows = 15000

# %% colab={"base_uri": "https://localhost:8080/"} id="o4h0u7cmV2oN" outputId="f8612dd7-4646-407f-e2f8-f376d89668b3"
completed = os.listdir(save_path)
if 'errors.txt' in completed:
  completed.remove('errors.txt')
completed_id = []
for element in completed:
  completed_id.append(element.replace('.pickle','').replace('kic ',''))
len(completed_id)

# %% colab={"base_uri": "https://localhost:8080/", "height": 354} id="0898e9f8" outputId="9aaab7eb-9291-4f4b-c916-d6547becdb4e"
period,epoch,disp=df[df['kepid']==kep_id][['koi_period','koi_time0bk','koi_disposition']].iloc[0]
period_2,epoch_2,disp_2=df[df['kepid']==kep_id_2][['koi_period','koi_time0bk','koi_disposition']].iloc[0]
print(period, epoch,disp)


# %% id="bbf9b665"
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
        file_name = path + 'kic '+str(self.kepler_id)+'-'+self.headers['Kepler_name']+'.pickle'
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

def apply_wavelet(light_curve,w_family, levels,cut_border_percent=0.1):
    time = light_curve.time.value
    data = light_curve.flux.value
    lc_wavelet = []
    for level in range(levels):
        level_w = pywt.dwt(data, w_family)
        lc_wavelet.append(cut_border(level_w,cut_border_percent))
        #lc_wavelet.append(level_w)
        data = level_w[0]
    return LightCurveWaveletFoldCollection(light_curve,lc_wavelet)

def load_light_curve(kepler_id,mission='Kepler'):
    kic = 'KIC '+str(kepler_id)
    lc_search = lk.search_lightcurve(kic, mission=mission)
    lc_collection = lc_search.download_all(download_dir="data/")
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

def cut_border(data_old,cut_percent=0.1):
    data_len_cut = int(len(data_old[0])*(cut_percent/2))
    data_new = [data[data_len_cut:(len(data)-data_len_cut)] for data in data_old ]
    return data_new
    
def process_light_curve(kepler_id,kepler_name,disp,period,epoch,w_family,levels,plot = False, plot_comparative=False,save=False, path="",wavelet_window=None,cut_border_percent=0.2):
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
    lc_w_par = apply_wavelet(lc_par,w_family,levels,cut_border_percent=cut_border_percent)
    # aplicamos wavelets a curvas inpar
    lc_w_inpar = apply_wavelet(lc_inpar,w_family,levels,cut_border_percent=cut_border_percent)
    headers = {
        "period": period,
        "epoch": epoch,
        "class": disp,
        "wavelet_family":w_family,
        "levels":levels,
        "window":wavelet_window,
        "border_cut":cut_border_percent,
        "Kepler_name":kepler_name
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

        koi_id,koi_name,disp, period, epoch=df_koi[['kepid','kepoi_name','koi_disposition','koi_period','koi_time0bk']].iloc[i]
        percent = i*100/(len(df_koi))
        print(f'procesando curva de luz KIC {int(koi_id)}-{koi_name}[{disp}] [{percent:.0f}%]')
        if not repeat_completed and (str(koi_id)+"-"+koi_name) in completed:
          print("curva de luz procesada anteriormente")
          continue
        try:
            process_light_curve(int(koi_id),koi_name,disp,period,epoch,wavelet_family,level,plot= plot,plot_comparative=plot_comparative,save = save_lc, path = save_path,wavelet_window=wavelet_windows)
            pass
        except:
            lc_errors.append(koi_id)
            print(f'Error with KIC {koi_id}')
    f = open (save_path+'errors.txt','w')
    for lc_error in lc_errors:
        text = 'KIC '+str(lc_error)+'\n'
        f.write(text)
    f.close()
    return lc_errors
    


# %% colab={"base_uri": "https://localhost:8080/", "height": 205} id="816dcfbc" outputId="fff90094-453c-44cb-8b72-5bfdfa354c3a"
result_window_out = process_light_curve(kep_id,"",disp,period,epoch,wavelet_family,level,plot_comparative=False)
result_window = process_light_curve(kep_id,"",disp,period,epoch,wavelet_family,level,plot_comparative=False,wavelet_window=15000)
result_window_2 = process_light_curve(kep_id_2,"",disp_2,period_2,epoch_2,wavelet_family,level,plot_comparative=False,wavelet_window=15000)

# %% id="5igZaXUfEdnh"
result_window.plot_comparative()
result_window_2.plot_comparative()

# %% id="p2ITN8cL05rb"
app_c_1_p = result_window.pliegue_par.get_approximation_coefficent(level=5)
app_c_2_p = result_window_2.pliegue_par.get_approximation_coefficent(level=5)

app_c_1_i = result_window.pliegue_inpar.get_approximation_coefficent(level=5)
app_c_2_i = result_window_2.pliegue_inpar.get_approximation_coefficent(level=5)

print(np.shape(app_c_1_p))
print(np.shape(app_c_2_p))
print(np.shape(app_c_1_i))
print(np.shape(app_c_2_i))
plt.plot(app_c_1_p,c='r')
plt.plot(app_c_1_i,c='b')
plt.show()
plt.plot(app_c_2_p,c='r')
plt.plot(app_c_2_i,c='b')

# %% [markdown] id="mZqsK9483IEu"
# # Ejecucion y trabajo sobre las curvas de luz en el dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="07c94f62" outputId="d0fe0b65-4aab-4905-b4f6-2fe091bafcf6"
errores = process_dataset(df,repeat_completed=False, completed=completed_id)

# %% colab={"base_uri": "https://localhost:8080/"} id="4hXwVSlmipvc" outputId="9568ae80-99d8-4c36-b2f8-4a907e34d950"
errores

# %% colab={"base_uri": "https://localhost:8080/"} id="mwDAjaCa7U7X" outputId="d25ef500-cb2c-4dcc-cf83-1c423256a0e1"
all_kep_id = df["kepid"]

result = dict({"completado":0,"faltante":0})
faltantes = []
for kep_id in all_kep_id:
  if str(kep_id) in completed_id:
    result["completado"]+=1
  else:
    result["faltante"]+=1
    faltantes.append(kep_id)

print(result)
print(faltantes)
len(completed_id)

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="9dc4ab5f" outputId="29eca306-fd0f-42fa-ed36-ab273d929eda"
path = save_path + completed[1]
print(path)
lcwC =  LightCurveWaveletCollection.load(path)
lcwC.headers
lcwC.plot_comparative()

# %% [markdown] id="ZUrpcLzfzKFL"
# # Visualizacion de curvas de luz

# %% colab={"base_uri": "https://localhost:8080/"} id="qgDDaX3D78_b" outputId="8fca7245-87fc-492e-ff20-fdc5ca0836f2"
selected = completed_id[2]
lc_name = 'kic '+str(selected)+'.pickle'
file_name = save_path+lc_name
lc = LightCurveWaveletCollection.load(file_name)
for i in range (0,10):
  lc_p = lc.pliegue_par.get_approximation_coefficent(level=i)
  lc_i = lc.pliegue_inpar.get_approximation_coefficent(level=i)
  print(f"[{i}]par: {len(lc_p)}, inpar: {len(lc_i)}")

# %% id="SGmklIJKpkNQ"
for lc_id in completed_id[:50]:
    lc_name = 'kic '+str(lc_id)+'.pickle'
    file_name = save_path+lc_name
    lc = LightCurveWaveletCollection.load(file_name)
    lc.plot_comparative()


# %% id="pLmwXTuGzI3q"
def plot_level_lc(level = 7 ,amount = 50):
  for lc_id in completed_id[:amount]:
    lc_name = 'kic '+str(lc_id)+'.pickle'
    file_name = save_path+lc_name
    lc = LightCurveWaveletCollection.load(file_name)
    lc_p = lc.pliegue_par.get_approximation_coefficent(level=level)
    lc_i = lc.pliegue_inpar.get_approximation_coefficent(level=level)
    plt.rcParams["figure.figsize"] = (20,3)
    plt.plot(lc_p,c="r")
    plt.plot(lc_i,c="b")
    plt.title(lc_name+":"+lc.headers['class'])
    plt.show()

def plot_level_lc_2(levels = [7] ,amount = 50):
  for lc_id in completed_id[:amount]:
    lc_name = 'kic '+str(lc_id)+'.pickle'
    file_name = save_path+lc_name
    lc = LightCurveWaveletCollection.load(file_name)
    lc_p =[ lc.pliegue_par.get_approximation_coefficent(level=level) for level in levels ]
    lc_i =[ lc.pliegue_inpar.get_approximation_coefficent(level=level) for level in levels ]
    
    ig, axarr = plt.subplots(nrows=1, ncols=len(levels), figsize=(26,26))
    for i in range(len(levels)):
      axarr[i].plot(lc_p,c='blue',label='par')
      axarr[i].plot(lc_i,c='blue',label='inpar')
      axarr[i].set_title("level:"+levels[i], fontsize=14)
    plt.title(lc_name+":"+lc.headers['class'])
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 318} id="SbmwOIPf1xy_" outputId="9e5e8d19-1214-4144-a925-62d363e33676"
plot_level_lc()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="SzilWyJj165I" outputId="f7a1a98e-6671-44ab-c8dc-b5e9ad20d671"
plot_level_lc(level=6)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="SiSt6eu12RNf" outputId="b2a7ab69-8144-4c69-edd0-4c3ddcf1a119"
plot_level_lc(level=8)
