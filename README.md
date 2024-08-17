# Versiones de los paquetes

```
python 3.7
cuda 11.2.2
cudnn 8.1.1.33
```

Para instalar primero intentarlo con requirements.in que es más flexible, y si no con requirements.txt que tiene todos los paquetes

# Notebooks

Están replicados los .py y los .ipynb, tienen el mismo código

01_descarga-Copy1.py y 01_descarga-Copy1.ipynb son para bajar en paralelo todas las curvas de luz y lo guarda en `download_dir`
Tarda ~50h en secuencial y ~8h en paralelo

binning.py y binning_test.py están sacados de https://github.com/google-research/exoplanet-ml/blob/master/exoplanet-ml/light_curve/ con alguna cosa añadida más para generar las funciones de `global_view` y `local_view` directamente. En la función `process_light_curve` de 01_descarga.py se ve cómo se llaman a estas funciones

01*descarga.py y 01_descarga.ipynb contienen el procesamiento completo para todo lo que hay descargado, también procesa en paralelo.
Crea un archivo `failure*{fecha actual}.csv`con las curvas que han fallado y la razón de fallo (excepción de Python), y el`download_dir`tiene que ser igual que el de 01_descarga.py-Copy1.py para que no se vuelvan a descargar las curvas de luz. Luego también en`path`guarda los pickles procesados que luego se pueden volver a cargar con`from_file`en el notebook 02_entrenamiento.py, y una carpeta`{path}/plot`donde guarda los plots de los pasos intermedios (mirar la docstring de`process_light_curve` de 01_descarga.py para más info)
Tarda ~10h en secuencial y ~3h en paralelo (con 16 procesadores y procesos)

LCWavelet.py contiene clases de apoyo para guardar y plotear curvas de luz de distintos tipos

