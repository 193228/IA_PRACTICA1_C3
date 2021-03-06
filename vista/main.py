# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1325, 868)
        MainWindow.setStyleSheet("background-color: rgb(28, 45, 63);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(660, 0, 20, 611))
        self.line.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setLineWidth(1)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 20, 231, 31))
        self.label.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 16pt \"MS Shell Dlg 2\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(760, 30, 521, 31))
        self.label_2.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 16pt \"MS Shell Dlg 2\";")
        self.label_2.setObjectName("label_2")
        self.pesoNumpy = QtWidgets.QLineEdit(self.centralwidget)
        self.pesoNumpy.setGeometry(QtCore.QRect(30, 180, 113, 22))
        self.pesoNumpy.setStyleSheet("color: rgb(255, 255, 255);")
        self.pesoNumpy.setText("")
        self.pesoNumpy.setObjectName("pesoNumpy")
        self.aprendizajeNumpy = QtWidgets.QLineEdit(self.centralwidget)
        self.aprendizajeNumpy.setGeometry(QtCore.QRect(270, 180, 113, 22))
        self.aprendizajeNumpy.setStyleSheet("color: rgb(255, 255, 255);")
        self.aprendizajeNumpy.setObjectName("aprendizajeNumpy")
        self.aprendizajeTensorFlow = QtWidgets.QLineEdit(self.centralwidget)
        self.aprendizajeTensorFlow.setGeometry(QtCore.QRect(770, 160, 113, 22))
        self.aprendizajeTensorFlow.setStyleSheet("color: rgb(255, 255, 255);")
        self.aprendizajeTensorFlow.setObjectName("aprendizajeTensorFlow")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 130, 121, 31))
        self.label_3.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.label_3.setObjectName("label_3")
        self.a = QtWidgets.QLabel(self.centralwidget)
        self.a.setGeometry(QtCore.QRect(200, 130, 251, 31))
        self.a.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.a.setObjectName("a")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(490, 130, 141, 31))
        self.label_5.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.label_5.setObjectName("label_5")
        self.umbralNumpy = QtWidgets.QLineEdit(self.centralwidget)
        self.umbralNumpy.setGeometry(QtCore.QRect(510, 180, 113, 22))
        self.umbralNumpy.setStyleSheet("color: rgb(255, 255, 255);")
        self.umbralNumpy.setObjectName("umbralNumpy")
        self.tablaNumpy = QtWidgets.QTableWidget(self.centralwidget)
        self.tablaNumpy.setGeometry(QtCore.QRect(40, 400, 581, 101))
        self.tablaNumpy.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"")
        self.tablaNumpy.setObjectName("tablaNumpy")
        self.tablaNumpy.setColumnCount(3)
        self.tablaNumpy.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        item.setBackground(QtGui.QColor(255, 255, 255))
        self.tablaNumpy.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tablaNumpy.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tablaNumpy.setHorizontalHeaderItem(2, item)
        self.tablaNumpy.horizontalHeader().setDefaultSectionSize(193)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(710, 110, 251, 31))
        self.label_6.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.label_6.setObjectName("label_6")
        self.tablaTensorFlow = QtWidgets.QTableWidget(self.centralwidget)
        self.tablaTensorFlow.setGeometry(QtCore.QRect(710, 360, 581, 101))
        self.tablaTensorFlow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.tablaTensorFlow.setObjectName("tablaTensorFlow")
        self.tablaTensorFlow.setColumnCount(3)
        self.tablaTensorFlow.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        item.setBackground(QtGui.QColor(255, 255, 255))
        self.tablaTensorFlow.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tablaTensorFlow.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tablaTensorFlow.setHorizontalHeaderItem(2, item)
        self.tablaTensorFlow.horizontalHeader().setDefaultSectionSize(193)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(750, 550, 551, 31))
        self.label_7.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.label_7.setObjectName("label_7")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(10, 610, 1341, 20))
        self.line_2.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setObjectName("line_2")
        self.botonNumpy = QtWidgets.QPushButton(self.centralwidget)
        self.botonNumpy.setGeometry(QtCore.QRect(220, 530, 221, 41))
        self.botonNumpy.setStyleSheet("font: 75 14pt \"MS Shell Dlg 2\";\n"
"color: rgb(255, 255, 255);")
        self.botonNumpy.setObjectName("botonNumpy")
        self.botonTensor = QtWidgets.QPushButton(self.centralwidget)
        self.botonTensor.setGeometry(QtCore.QRect(870, 490, 221, 41))
        self.botonTensor.setStyleSheet("font: 75 14pt \"MS Shell Dlg 2\";\n"
"color: rgb(255, 255, 255);")
        self.botonTensor.setObjectName("botonTensor")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(190, 300, 291, 31))
        self.label_4.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.label_4.setObjectName("label_4")
        self.generacionesNumpy = QtWidgets.QLineEdit(self.centralwidget)
        self.generacionesNumpy.setGeometry(QtCore.QRect(270, 350, 113, 22))
        self.generacionesNumpy.setStyleSheet("color: rgb(255, 255, 255);")
        self.generacionesNumpy.setObjectName("generacionesNumpy")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(860, 260, 291, 31))
        self.label_8.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.label_8.setObjectName("label_8")
        self.generacionesTensorflow = QtWidgets.QLineEdit(self.centralwidget)
        self.generacionesTensorflow.setGeometry(QtCore.QRect(940, 310, 113, 22))
        self.generacionesTensorflow.setStyleSheet("color: rgb(255, 255, 255);")
        self.generacionesTensorflow.setObjectName("generacionesTensorflow")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(1050, 110, 141, 31))
        self.label_9.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.label_9.setObjectName("label_9")
        self.umbralTensorflow = QtWidgets.QLineEdit(self.centralwidget)
        self.umbralTensorflow.setGeometry(QtCore.QRect(1060, 150, 113, 22))
        self.umbralTensorflow.setStyleSheet("color: rgb(255, 255, 255);")
        self.umbralTensorflow.setObjectName("umbralTensorflow")
        self.mejorTazaNativo = QtWidgets.QRadioButton(self.centralwidget)
        self.mejorTazaNativo.setGeometry(QtCore.QRect(230, 220, 201, 20))
        self.mejorTazaNativo.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 10pt \"MS Shell Dlg 2\";\n"
"")
        self.mejorTazaNativo.setCheckable(True)
        self.mejorTazaNativo.setChecked(True)
        self.mejorTazaNativo.setAutoExclusive(False)
        self.mejorTazaNativo.setObjectName("mejorTazaNativo")
        self.pesoAleatorio = QtWidgets.QRadioButton(self.centralwidget)
        self.pesoAleatorio.setGeometry(QtCore.QRect(40, 220, 101, 20))
        self.pesoAleatorio.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 10pt \"MS Shell Dlg 2\";\n"
"")
        self.pesoAleatorio.setCheckable(True)
        self.pesoAleatorio.setChecked(False)
        self.pesoAleatorio.setAutoExclusive(False)
        self.pesoAleatorio.setObjectName("pesoAleatorio")
        self.mejorTazaLibreria = QtWidgets.QRadioButton(self.centralwidget)
        self.mejorTazaLibreria.setGeometry(QtCore.QRect(730, 200, 201, 20))
        self.mejorTazaLibreria.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 10pt \"MS Shell Dlg 2\";\n"
"")
        self.mejorTazaLibreria.setCheckable(True)
        self.mejorTazaLibreria.setChecked(True)
        self.mejorTazaLibreria.setAutoExclusive(False)
        self.mejorTazaLibreria.setObjectName("mejorTazaLibreria")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(40, 720, 1171, 41))
        self.label_10.setObjectName("label_10")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1325, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Perceptron Nativo"))
        self.label_2.setText(_translate("MainWindow", "Perceptron Utilizando Tensorflow (libreria)"))
        self.label_3.setText(_translate("MainWindow", "Ingrese peso"))
        self.a.setText(_translate("MainWindow", "Ingrese taza de aprendizaje"))
        self.label_5.setText(_translate("MainWindow", "Ingrese umbral"))
        item = self.tablaNumpy.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ETA"))
        item = self.tablaNumpy.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Pesos Finales"))
        item = self.tablaNumpy.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Norma Error"))
        self.label_6.setText(_translate("MainWindow", "Ingrese taza de aprendizaje"))
        item = self.tablaTensorFlow.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ETA"))
        item = self.tablaTensorFlow.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Pesos Finales"))
        item = self.tablaTensorFlow.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Norma Error"))
        self.label_7.setText(_translate("MainWindow", "El peso se genera aleatoriamente en un rango de -2 a 2"))
        self.botonNumpy.setText(_translate("MainWindow", "Evaluar Y Graficar"))
        self.botonTensor.setText(_translate("MainWindow", "Evaluar Y Graficar"))
        self.label_4.setText(_translate("MainWindow", "Ingrese Generaciones Maximas"))
        self.label_8.setText(_translate("MainWindow", "Ingrese Generaciones Maximas"))
        self.label_9.setText(_translate("MainWindow", "Ingrese umbral"))
        self.mejorTazaNativo.setText(_translate("MainWindow", "Mejor Taza (0.000001)"))
        self.pesoAleatorio.setText(_translate("MainWindow", "Aleatorio"))
        self.mejorTazaLibreria.setText(_translate("MainWindow", "Mejor Taza (0.00001)"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#ffffff;\">El Dataset Evaluado Se Puede Encontrar En El Siguiente Enlace: https://drive.google.com/file/d/1z9MtcLFBp_DyLdcYMHrxxaV381YhaQSX/view?usp=sharing</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
