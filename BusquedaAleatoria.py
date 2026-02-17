import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Funciones auxiliares

def calcular_MSE_segmento(puntos_segmento):
    puntos_segmento = np.array(puntos_segmento)
    n = len(puntos_segmento)
    X = np.arange(n).reshape(-1, 1) # Para ponerlo en forma de columna y que lo pille sklearn
    y = puntos_segmento.reshape(-1, 1)

    modelo = LinearRegression()
    modelo.fit(X, y)
    prediccion = modelo.predict(X)

    mse = mean_squared_error(y, prediccion)

    return mse

def calcular_RMSE(serie_temporal, limites_segmentos):
    Ndatos = len(serie_temporal)
    serie_completa = [0] + limites_segmentos + [Ndatos]
    errores_segmento = []

    for i in range(len(serie_completa) - 1):
        inicio = serie_completa[i]
        fin = serie_completa[i+1]

        segmento = serie_temporal[inicio:fin]
        mse_segmento = calcular_MSE_segmento(segmento)
        errores_segmento.append(mse_segmento)

    mse_mean = np.mean(errores_segmento)

    return mse_mean

def readSeries(filename) -> list: # Función para crear la lista que define la serie
    data = np.loadtxt(filename).tolist()
    return data

def getBreakingPoints(n_points, k_segments): # Función de búsqueda aleatoria
    
    breaking_points = [0,n_points-1]

    for _ in range(k_segments -1):
        n = random.randint(0,n_points-1)
        
        while n in breaking_points:
            n = random.randint(0,n_points-1)
        
        breaking_points.append(n)
    
    breaking_points.sort()

    return breaking_points

def randomSearch(series: list, k_segments):
