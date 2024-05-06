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

# %% id="cb1ccdb1" editable=true slideshow={"slide_type": ""}
import pandas as pd

# %% [markdown] id="1c3b8c7c"
# https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

# %% id="5aa5e5bc"
df_light_curves = pd.read_csv('cumulative_2022.09.30_09.06.43.csv',skiprows=144)

# %% [markdown] id="738dc38a"
# - COLUMN kepid:          KepID
# - COLUMN kepoi_name:     KOI Name
# - COLUMN kepler_name:    Kepler Name
# - COLUMN koi_disposition: Exoplanet Archive Disposition
# - COLUMN koi_pdisposition: Disposition Using Kepler Data
# - COLUMN koi_score:      Disposition Score
# - COLUMN koi_fpflag_nt:  Not Transit-Like False Positive Flag
# - COLUMN koi_fpflag_ss:  Stellar Eclipse False Positive Flag
# - COLUMN koi_fpflag_co:  Centroid Offset False Positive Flag
# - COLUMN koi_fpflag_ec:  Ephemeris Match Indicates Contamination False Positive Flag
# - COLUMN koi_period:     Orbital Period [days]
# - COLUMN koi_period_err1: Orbital Period Upper Unc. [days]
# - COLUMN koi_period_err2: Orbital Period Lower Unc. [days]
# - COLUMN koi_time0bk:    Transit Epoch [BKJD]
# - COLUMN koi_time0bk_err1: Transit Epoch Upper Unc. [BKJD]
# - COLUMN koi_time0bk_err2: Transit Epoch Lower Unc. [BKJD]
# - COLUMN koi_impact:     Impact Parameter
# - COLUMN koi_impact_err1: Impact Parameter Upper Unc.
# - COLUMN koi_impact_err2: Impact Parameter Lower Unc.
# - COLUMN koi_duration:   Transit Duration [hrs]
# - COLUMN koi_duration_err1: Transit Duration Upper Unc. [hrs]
# - COLUMN koi_duration_err2: Transit Duration Lower Unc. [hrs]
# - COLUMN koi_depth:      Transit Depth [ppm]
# - COLUMN koi_depth_err1: Transit Depth Upper Unc. [ppm]
# - COLUMN koi_depth_err2: Transit Depth Lower Unc. [ppm]
# - COLUMN koi_prad:       Planetary Radius [Earth radii]
# - COLUMN koi_prad_err1:  Planetary Radius Upper Unc. [Earth radii]
# - COLUMN koi_prad_err2:  Planetary Radius Lower Unc. [Earth radii]
# - COLUMN koi_teq:        Equilibrium Temperature [K]
# - COLUMN koi_teq_err1:   Equilibrium Temperature Upper Unc. [K]
# - COLUMN koi_insol:      Insolation Flux [Earth flux]
# - COLUMN koi_insol_err1: Insolation Flux Upper Unc. [Earth flux]
# - COLUMN koi_insol_err2: Insolation Flux Lower Unc. [Earth flux]
# - COLUMN koi_model_snr:  Transit Signal-to-Noise
# - COLUMN koi_tce_plnt_num: TCE Planet Number
# - COLUMN koi_tce_delivname: TCE Delivery
# - COLUMN koi_steff:      Stellar Effective Temperature [K]
# - COLUMN koi_steff_err1: Stellar Effective Temperature Upper Unc. [K]
# - COLUMN koi_steff_err2: Stellar Effective Temperature Lower Unc. [K]
# - COLUMN koi_slogg:      Stellar Surface Gravity [log10(cm/s**2)]
# - COLUMN koi_slogg_err1: Stellar Surface Gravity Upper Unc. [log10(cm/s**2)]
# - COLUMN koi_slogg_err2: Stellar Surface Gravity Lower Unc. [log10(cm/s**2)]
# - COLUMN koi_srad:       Stellar Radius [Solar radii]
# - COLUMN koi_srad_err1:  Stellar Radius Upper Unc. [Solar radii]
# - COLUMN koi_srad_err2:  Stellar Radius Lower Unc. [Solar radii]
# - COLUMN ra:             RA [decimal degrees]
# - COLUMN dec:            Dec [decimal degrees]
# - COLUMN koi_kepmag:     Kepler-band [mag]

# %% id="62ff2707" outputId="835bff8e-db0e-4fef-eca6-b6f4ae79be56"
df_light_curves

# %% id="1c3921b7" outputId="76aaaa53-caaa-413e-bf07-1842b7219b5a"
# datos que nos sirven para clasificar la estrella
df = df_light_curves[['kepid','kepoi_name','koi_disposition','koi_steff','koi_srad','koi_smass']]
df

# %% id="3c674545" outputId="c4c1a14a-ced5-40ff-8488-e31006df1012"
# revision de valores nullos en los datos de filtro
df.isnull().sum()

# %% id="2373f26c" outputId="303326a2-0fd6-4b27-803a-fc09fbf549d3"
# evaluacion rapida de los datos
df.describe()

# %% [markdown] id="4be9a8e1"
# Caracteristicas de cada estrella (Segun wikipedia)

# %% [markdown] id="cfb018c9"
# <img src="stars_type.png">

# %% id="b141aed8"
O_stars ={
    "max_temp" : 10000000000000, 
    "min_temp" : 33000,
    "max_masa" : 1000,
    "min_masa" : 16,
    "max_rad" : 1000,
    "min_rad" : 6.6,
    "max_light" : 1000000,
    "min_light" : 30000
}

B_stars ={
    "max_temp" : 33000,
    "min_temp" : 10000,
    "max_masa" : 16,
    "min_masa" : 2.1,
    "max_rad" : 6.6,
    "min_rad" : 1.8,
    "max_light" : 30000,
    "min_light" : 25
}

A_stars ={
    "max_temp" : 10000, 
    "min_temp" : 7500, 
    "max_masa" : 2.1, 
    "min_masa" : 1.4, 
    "max_rad" : 1.8,
    "min_rad" : 1.4,
    "max_light" : 25,
    "min_light" : 5
}

F_stars ={
    "max_temp" : 7500,
    "min_temp" : 6000,
    "max_masa" : 1.4,
    "min_masa" : 1.04,
    "max_rad" : 1.4,
    "min_rad" : 1.15,
    "max_light" : 5,
    "min_light" : 1.5
}

G_stars ={
    "max_temp" : 6000,
    "min_temp" : 5200,
    "max_masa" : 1.04,
    "min_masa" : 0.8,
    "max_rad" : 1.15,
    "min_rad" : 0.96,
    "max_light" : 1.5,
    "min_light" : 0.6
}

K_stars ={
    "max_temp" : 5200, 
    "min_temp" : 3700, 
    "max_masa" : 0.8 , 
    "min_masa" : 0.45, 
    "max_rad" : 0.96 , 
    "min_rad" : 0.7, 
    "max_light" : 0.6,
    "min_light" : 0.08
}

M_stars ={
    "max_temp" : 3700, 
    "min_temp" : 0, 
    "max_masa" : 0.45, 
    "min_masa" : 0, 
    "max_rad" : 0.7, 
    "min_rad" : 0,
    "max_light" : 0.08,
    "min_light" : 0
}


# %% id="43f73cc2"
def filter_data (df ,star):
    """
    Funcion para filtrar los registros segun las caracteristicas de las estrellas que se pasen.
    
    Parametros
    -------------------
    df: pandas.Dataframe
        Dataframe desde donde se extraen los registros de cada curva de luz.
    star: dict
        diccionario con los valores maximos y minimos de las estrellas para su filtro.
        
    Retorna
    -------------------
    df: pandas.Dataframe
        Dataframe con los registros que cumplen las condiciones maximas y minimas entregadas por el diccionario.
    """
    return df[ 
        # radio de la estrella
        ((df['koi_srad']<= star["max_rad"]) & (df['koi_srad'] >= star["min_rad"]) )
        #temperatura de la estrella (agregar '&' en caso de activar la anterior)
        &     ( (df['koi_steff']<= star["max_temp"]) & (df['koi_steff']>= star["min_temp"]) )
        #masa de la estrella
        #      & ( (df['koi_smass']<= star["max_masa"]) & (df['koi_smass']>= star["min_masa"]) )
    ]

def dataset_stadistic(df):
    """
    Entrega datos estadisticos de los registros:
    - Cantidad de registros.
    - Cantidad de planetas confirmados.
    - Cantidad de falsos positivos.
    - Cantidad de planetas candidatos.
    - Balanceo: carga de los datos entre confirmados y Falsos positivos que varia entre -1 y 1.
                  - -1: todos los valores son falsos positivos
                  -  0: misma cantidad de planetas confirmados y falsos positivos.
                  -  1: todos los valores son planetas confirmados.
    
    Parametros
    -------------------
    df: pandas.Dataframe
        Dataframe desde donde se extraen los registros de cada curva de luz.
    star: dict
        diccionario con los valores maximos y minimos de las estrellas para su filtro.
        
    Retorna
    -------------------
    df: pandas.Dataframe
        Dataframe con los registros que cumplen las condiciones maximas y minimas entregadas por el diccionario.
    """
    stadistic = []
    
    stadistic.append({'metrica':'Cantidad total de registros','valores':len(df)},)
    stadistic.append({'metrica':'Planetas confirmados','valores':len(df[df['koi_disposition']=='CONFIRMED' ])})
    stadistic.append({'metrica':'Planetas candidatos','valores':len(df[df['koi_disposition']=='CANDIDATE' ])})
    stadistic.append({'metrica':'Falso positivo','valores':len(df[df['koi_disposition']=='FALSE POSITIVE' ])})
    try:
        balanceo = (len(df[df['koi_disposition']=='CONFIRMED' ]) - len(df[df['koi_disposition']=='FALSE POSITIVE' ]))/(len(df[df['koi_disposition']=='CONFIRMED' ])+len(df[df['koi_disposition']=='FALSE POSITIVE' ]))
    except:
        balanceo = 'Error'
    stadistic = stadistic.append({'metrica':'balanceo','valores':balanceo})
    # pd.set_option('precision', 2)
    display(pd.DataFrame(stadistic))


# %% id="a5a19090" outputId="8d2f0781-df87-45a2-a90f-f8207aa8fd7c"
df_m_stars = filter_data(df,M_stars)
df_k_stars =filter_data(df,K_stars)
df_g_stars =filter_data(df,G_stars)
df_f_stars =filter_data(df,F_stars)
df_a_stars =filter_data(df,A_stars)
df_b_stars =filter_data(df,B_stars)
df_o_stars =filter_data(df,O_stars)

print('Estrellas tipo M')
dataset_stadistic(df_m_stars)

print('\nEstrellas tipo K')
dataset_stadistic(df_k_stars)

print('Estrellas tipo G')
dataset_stadistic(df_g_stars)

print('\nEstrellas tipo F')
dataset_stadistic(df_f_stars)

print('Estrellas tipo A')
dataset_stadistic(df_a_stars)

print('\nEstrellas tipo B')
dataset_stadistic(df_b_stars)

print('\nEstrellas tipo O')
dataset_stadistic(df_o_stars)

# %% id="6b8917df"
df_save_M=filter_data(df_light_curves,M_stars)
df_save_K=filter_data(df_light_curves,K_stars)
df_save_G=filter_data(df_light_curves,G_stars)

# %% id="755445f4"
df_save_M_filtered = df_save_M[(df_save_M['koi_disposition']== 'CONFIRMED') | (df_save_M['koi_disposition']== 'FALSE POSITIVE') ]

df_save_K_filtered = df_save_K[(df_save_K['koi_disposition']== 'CONFIRMED') | (df_save_K['koi_disposition']== 'FALSE POSITIVE') ]
df_save_G_filtered = df_save_G[(df_save_G['koi_disposition']== 'CONFIRMED') | (df_save_G['koi_disposition']== 'FALSE POSITIVE') ]

# %% id="037c254a"
df_save_M.to_csv('light_curves_M_stars.csv',index=False)
df_save_K.to_csv('light_curves_K_stars.csv',index=False)
df_save_G.to_csv('light_curves_G_stars.csv',index=False)

df_save_M_filtered.to_csv('light_curves_M_stars_filter.csv',index=False)
df_save_K_filtered.to_csv('light_curves_K_stars_filter.csv',index=False)
df_save_G_filtered.to_csv('light_curves_G_stars_filter.csv',index=False)
