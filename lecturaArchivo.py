import pandas as pd

def leerArchivo():
    eleccion = "dataset_c3.csv"
    x = pd.read_csv(eleccion)
    return x