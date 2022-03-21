import sys
import PyQt5
import numpy as np
from PyQt5 import QtWidgets

from lecturaArchivo import leerArchivo
from vista.main import Ui_MainWindow as ventanaPrincipal
from sinLibreria.ia_train import ejecutarAlgoritmoSinLibreria
from libreria.ia_tensor import ejecutarAlgoritmoLibreria

class MyApp(PyQt5.QtWidgets.QMainWindow, ventanaPrincipal):
    def __init__(self):
        PyQt5.QtWidgets.QMainWindow.__init__(self)
        ventanaPrincipal.__init__(self)
        self.setupUi(self)
        lecturaDataset = leerArchivo()
        lecturaX = lecturaDataset['X'].tolist()
        X = transformX(lecturaX) #cargo al iniciar los valores de x
        Y = lecturaDataset['Y'].tolist() #cargo al iniciar los valores de y
        acciones(self,X,Y)

def acciones(ventana,X,Y):
    ventana.botonNumpy.clicked.connect(lambda: algoritmoSinLibreria(ventana,X,Y))
    ventana.botonTensor.clicked.connect(lambda: algoritmoConLibreria(ventana,X,Y))
    #ventana.

def algoritmoSinLibreria(ventana,X,Y):
    valores = ajusteNativo(ventana)
    datos = ejecutarAlgoritmoSinLibreria(valores['peso'],valores['eta'],valores['umbral'],X,Y,valores['generacionesMaximas']) #peso se le agrega
    tablaSinLibreria(ventana,datos)

def algoritmoConLibreria(ventana,X,Y):
    valores = ajusteLibreria(ventana)
    datos = ejecutarAlgoritmoLibreria(X,Y,valores['generacionesMaximas'],valores['umbral'],valores['eta'])
    tablaLibreria(ventana,datos)

def ajusteNativo(ventana):
    if ventana.pesoAleatorio.isChecked():
        peso = np.random.uniform(-2,2)
    else:
        peso = float(ventana.pesoNumpy.text())

    if ventana.mejorTazaNativo.isChecked():
        eta = 0.000001
    else:
        eta = float(ventana.aprendizajeNumpy.text())

    diccionario = {
        "peso": [peso],
        "eta": eta,
        "umbral": float(ventana.umbralNumpy.text()),
        "generacionesMaximas": int(ventana.generacionesNumpy.text())
    }
    return diccionario

def ajusteLibreria(ventana):
    if ventana.mejorTazaLibreria.isChecked():
        eta = 0.00001
    else:
        eta = float(ventana.aprendizajeTensorFlow.text())

    diccionario = {
        "eta": eta,
        "umbral": float(ventana.umbralTensorflow.text()),
        "generacionesMaximas": int(ventana.generacionesTensorflow.text())
    }
    return diccionario

def tablaLibreria(ventana,lista):
    ventana.tablaTensorFlow.setRowCount(1)
    ventana.tablaTensorFlow.setItem(0,0,QtWidgets.QTableWidgetItem(str(lista["eta"])))
    ventana.tablaTensorFlow.setItem(0,1,QtWidgets.QTableWidgetItem(str(lista["pesoFinal"])))
    ventana.tablaTensorFlow.setItem(0,2,QtWidgets.QTableWidgetItem(str(lista["normaError"])))

def tablaSinLibreria(ventana,lista):
    ventana.tablaNumpy.setRowCount(1)
    ventana.tablaNumpy.setItem(0, 0, QtWidgets.QTableWidgetItem(str(lista["eta"].iloc[-1])))
    ventana.tablaNumpy.setItem(0, 1, QtWidgets.QTableWidgetItem(str(lista["wk"].iloc[-1])))
    ventana.tablaNumpy.setItem(0, 2, QtWidgets.QTableWidgetItem(str(lista["norma"].iloc[-1])))


def transformX(x):
    X = []
    for i in x:
        X.append([i])
    return X

if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)  # crea un objeto de aplicación (Argumentos de sys)
    window = MyApp()
    window.show()
    window.setFixedSize(window.size())
    app.exec_()