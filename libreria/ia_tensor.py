import keras.layers
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from lecturaArchivo import leerArchivo

class umbral:
    def get_umbral(self):
        return self._umbral
    def set_umbral(self, x):
        self._umbral = x

class evaluarUmbral(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= umb.get_umbral()):
            print("termino")
            self.model.stop_training = True

umb = umbral()

def ejecutarAlgoritmoLibreria(X,Y,generaciones,umbral,eta):
    umb.set_umbral(umbral)
    detenerEntrenamiento = evaluarUmbral()
    capa = keras.layers.Dense(units=1, input_shape=[1],activation='linear')  # 1 neurona de 1 dimension units numero de capas, el cual genera randoms de 0 a 0.01
    modelo = keras.Sequential([capa])
    modelo.compile(
        optimizer=keras.optimizers.SGD(learning_rate=eta), # la mejor eta es de 0.00001 para el error minimo
        #optimizer=keras.optimizers.SGD(learning_rate=0.00001),  # cambiamos el optimizador por el que mejor se adecuo a nuestra practica
        loss='mean_squared_error',  #
    )
    historial = modelo.fit(X, Y, epochs=generaciones, verbose=True, callbacks=[detenerEntrenamiento])
    print(modelo.get_weights())

    pesoFinal = modelo.get_weights()[1][0]
    normaError = historial.history['loss'][-1]
    diccionario = {
        "pesoFinal": pesoFinal,
        "eta":eta,
        "normaError":normaError
    }


    plt.xlabel('Generacion')
    plt.ylabel('Error')
    plt.plot(historial.history['loss'])
    plt.show()

    return diccionario

