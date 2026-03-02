import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random
import statistics
import math
import os

# Función para generar puntos de corte aleatorios
def getBreakingPoints(n_points, k_segments): 
    breaking_points = [0,n_points-1]

    for _ in range(k_segments -1):
        n = random.randint(0,n_points-1)
        while n in breaking_points:
            n = random.randint(0,n_points-1)
        breaking_points.append(n)
    breaking_points.sort()

    return breaking_points

#Funcion para limpiar pantalla? codigos ASCII peruanillos
def clear_screen():
    print('\033[2J\033[H', end='')

# Función para seleccionar la serie para hacer la regresión
def select_series():
    series_dict = {
        "TS1.txt": 9,"TS2.txt": 10,"TS3.txt": 20,"TS4.txt": 50
    }
    print("Tenemos estas series con sus k segmentos: ")
    for i, name in enumerate(series_dict.keys(), 1):
        print(f"{i}. {name} ({series_dict[name]} segmentos)")

    while True:
        try:
            opcion = int(input("Introduce serie (1-4): "))
            if 1 <= opcion <= len(series_dict):
                filename = list(series_dict.keys())[opcion-1]
                k_segments = series_dict[filename]
                return filename, k_segments
            else:
                print("Opcion invalida crack.")
        except ValueError:
            print("Pon un numero que sirva.")

# Función para crear la lista que define la serie
def readSeries(filename) -> list: 
    data = np.loadtxt(filename).tolist()
    return data

#Funcion graficar la serie
def draw(Y,breaking_points, filename, title="Regresión por Segmentos"):
    X=list(range(len(Y)))
    plt.plot(X,Y,color='blue',label='Serie')

    #Marcar segmentos
    if breaking_points:
        X_cuts = breaking_points
        Y_cuts = [Y[i] for i in breaking_points]
        plt.scatter(X_cuts, Y_cuts, color='red', s=50, label='Puntos de corte', zorder=5)

        # Dibujar las rectas de regresión por segmento
        for i in range(len(breaking_points)-1):
            start = breaking_points[i]
            end = breaking_points[i+1]

            # Crear X segmentada y reshape para LinearRegression
            X_seg = np.arange(start, end).reshape(-1, 1)
            y_seg = np.array(Y[start:end]).reshape(-1, 1)

            model = LinearRegression()
            model.fit(X_seg, y_seg)
            y_pred = model.predict(X_seg)

            # Dibujar la recta de regresión
            plt.plot(X_seg, y_pred, color='orange', linewidth=2, label='Regresión' if i==0 else None)

    plt.title(filename)
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.grid(True)
    #plt.legend(loc='upper left', bbox_to_anchor=(1,1))  # Fuera a la derecha
    plt.show()


def segmentMSE(breaking_points):
    breaking_points = np.array(breaking_points)
    n = len(breaking_points)
    X = np.arange(n).reshape(-1, 1) # Para ponerlo en forma de columna y que lo pille sklearn
    y = breaking_points.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X)

    mse = mean_squared_error(y, prediction)

    return mse

def avgMSE(temp_series, breaking_points):
    segment_errs = []

    for i in range(len(breaking_points) - 1):
        start = breaking_points[i]
        end = breaking_points[i+1]

        segment = temp_series[start:end]
        segment_mse = segmentMSE(segment)
        segment_errs.append(segment_mse)

    mse_mean = np.mean(segment_errs)

    return mse_mean

def calculateVariance(data):
    return statistics.variance(data)

def calculateStandardDesviation(data):
    return math.sqrt(calculateVariance(data))

def calculateErrorMean(data):
    data = np.array(data)
    return np.mean(data)

def save_statistics(filename_log, method_name, series_name, k, exec_time, mse, avg_error, variance, std_dev):
    """
    Guarda los resultados de la ejecución en un archivo CSV por columnas.
    """
    file_exists = os.path.isfile(filename_log)
    
    with open(filename_log, mode='a', encoding='utf-8') as f:
        # Si el archivo es nuevo, escribimos la cabecera
        if not file_exists:
            header = "Metodo,Serie,K,Tiempo_s,MSE,Avg_Error,Varianza,Std_Dev\n"
            f.write(header)
        
        # Escribimos los datos
        line = f"{method_name},{series_name},{k},{exec_time:.6f},{mse:.6f},{avg_error:.6f},{variance:.6f},{std_dev:.6f}\n"
        f.write(line)
    print(f"--> Estadísticas guardadas en: {filename_log}")