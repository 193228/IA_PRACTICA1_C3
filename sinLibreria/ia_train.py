import pandas as pd
from matplotlib import pyplot as plt
from sinLibreria.metodos import *


def ejecutarAlgoritmoSinLibreria(peso,eta,umbral,X,Y,maxIter):
    r = entrenamiento(peso, eta, umbral, X, Y, maxIter)
    lista = grafIteracion(r)
    return lista

def entrenamiento(Wcero, eta, umbral, X, Y, maxIter):
    iteracion = 0
    l = []
    while True:
        Wk = []
        if (iteracion == 0):
            Wk = Wcero
        else:
            Wk = wk.copy()
        Uk = calcularU(X, Wk)
        yc = yCalculada(Uk)
        error = calcularError(Y, yc)
        ex = multiplicarEX(error, X)
        p = multiplicacionEta(eta, ex)
        wk = nuevoW(Wk, p)
        norma = np.linalg.norm(error)
        dic = {
            "iteracion": iteracion,
            "norma": norma,
            "wk": Wk,
            "eta": eta
        }
        l.append(dic)
        iteracion += 1
        try:
            if norma < umbral or iteracion == maxIter:
                break
            else:
                print("norma no es menor a umbral: ", "norma es: ", norma, " umbral es: ", umbral)
        except:
            print("Ocurrio un ciclo")
    return l

def grafIteracion(lista):
    df = pd.DataFrame(lista)
    fig, ax = plt.subplots()
    ax.plot(df.index.values, df["norma"])
    plt.xlabel('Iteraciones')  # override the xlabel
    plt.ylabel('Norma Error')  # override the ylabel
    plt.title('Grafica de norma de error eta: '+ str(df["eta"][0]))  # override the title
    plt.show()
    return df