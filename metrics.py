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
def draw(Y, breaking_points, filename, title="Regresión por Segmentos"):
    X = list(range(len(Y)))
    plt.plot(X, Y, color='blue', label='Serie')

    # Marcar puntos de corte con líneas verticales discontinuas
    if breaking_points:
        # Obtener el rango del eje Y para las líneas verticales
        y_min, y_max = plt.ylim()
        
        for bp in breaking_points:
            plt.axvline(x=bp, color='red', linestyle='--', linewidth=1, alpha=0.7, 
                       label='Puntos de corte' if bp == breaking_points[0] else None)

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
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))  # Fuera a la derecha
    plt.show()

#Funcion hallar MSE de un segmento respecto a un intervalo de la serie
def segmentMSE(segment):
    segment = np.array(segment)
    n = len(segment)
    X = np.arange(n).reshape(-1, 1) # Para ponerlo en forma de columna y que lo pille sklearn
    y = segment.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X)

    mse = mean_squared_error(y, prediction)

    return mse

#Funcion que hace la media de los MSE de todos los intervalos
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

#Funcion hallar varianza
def calculateVariance(data):
    return statistics.variance(data)

#Funcion hallar desviacion tipica
def calculateStandardDesviation(data):
    return math.sqrt(calculateVariance(data))

#Funcion hallar media del error
def calculateErrorMean(data):
    data = np.array(data)
    return np.mean(data)

#Funcion para guardar las metricas en un CSV
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

# Añade esto al final de metrics.py
def draw_multiple_stats_regression(ejecuciones, stats_dict, algorithm_name, filename_series):
    X = np.array(ejecuciones).reshape(-1, 1)
    num_stats = len(stats_dict)
    
    fig, axs = plt.subplots(num_stats, 1, figsize=(10, 5 * num_stats))
    
    # Si solo hay 1 estadística, axs no es un array, lo convertimos a lista
    if num_stats == 1:
        axs = [axs]
        
    for ax, (stat_name, Y) in zip(axs, stats_dict.items()):
        Y_arr = np.array(Y).reshape(-1, 1)
        
        # Modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, Y_arr)
        y_pred = model.predict(X)
        
        ax.scatter(X, Y, color='blue', alpha=0.6, label=f'{stat_name} real')
        ax.plot(X, y_pred, color='red', linewidth=2, label='Tendencia (Regresión)')
        
        ax.set_title(f"Evolución de {stat_name} - {algorithm_name} ({filename_series})")
        ax.set_xlabel("Número de Ejecución")
        ax.set_ylabel(stat_name)
        ax.grid(True)
        ax.legend()
        
    plt.tight_layout()
    plt.show()
    
def draw_iterations_study(iteraciones_list, mses_medios, nombre, filename):
    X = np.array(iteraciones_list).reshape(-1, 1)
    Y = np.array(mses_medios).reshape(-1, 1)
    
    # Regresión lineal para ver la tendencia
    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, marker='o', color='blue', linewidth=2, label='MSE Medio Real')
    plt.plot(X, y_pred, color='red', linestyle='--', label='Tendencia (Regresión)')
    
    plt.title(f"Convergencia: MSE Medio vs N.º de Iteraciones\n{nombre} ({filename})")
    plt.xlabel("Número Máximo de Iteraciones")
    plt.ylabel("MSE Medio")
    plt.grid(True)
    plt.legend()
    plt.show()