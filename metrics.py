from enum import show_flag_values
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
import math
import os

# --- LÓGICA DE SEGMENTACIÓN ---

def getBreakingPoints(n_points, k_segments): 
    breaking_points = [0, n_points-1]
    for _ in range(k_segments - 1):
        n = random.randint(0, n_points-1)
        while n in breaking_points:
            n = random.randint(0, n_points-1)
        breaking_points.append(n)
    breaking_points.sort()
    return breaking_points

def clear_screen():
    print('\033[2J\033[H', end='')

def select_series():
    series_dict = {"TS1.txt": 9, "TS2.txt": 10, "TS3.txt": 20, "TS4.txt": 50}
    print("\nSeries disponibles para segmentar:")
    for i, name in enumerate(series_dict.keys(), 1):
        print(f"{i}. {name} ({series_dict[name]} segmentos)")
    while True:
        try:
            opcion = int(input("Seleccione serie (1-4): "))
            if 1 <= opcion <= len(series_dict):
                filename = list(series_dict.keys())[opcion-1]
                return filename, series_dict[filename]
            print("Opción inválida.")
        except ValueError:
            print("Introduzca un número válido.")

def readSeries(filename): 
    return np.loadtxt(filename).tolist()

def segmentMSE(segment):
    segment = np.array(segment)
    n = len(segment)
    if n < 2: return 0.0
    x = np.arange(n)
    x_matrix = np.vstack([x, np.ones(n)]).T
    _, ssr, _, _ = np.linalg.lstsq(x_matrix, segment, rcond=None)
   
    if len(ssr) > 0:
        return ssr[0] / n
    
    else:
        return 0.0

def avgMSE(temp_series, breaking_points):
    segment_errs = []
    for i in range(len(breaking_points) - 1):
        segment = temp_series[breaking_points[i]:breaking_points[i+1]]
        segment_errs.append(segmentMSE(segment))
    return np.mean(segment_errs)

# --- ESTADÍSTICAS Y PERSISTENCIA ---

def calculateVariance(data):
    if len(data) > 1:
        return statistics.variance(data) 
    else:
        return 0.0

def calculateStandardDesviation(data):
    return math.sqrt(calculateVariance(data))

def calculateErrorMean(data):
    return np.mean(data)

def save_statistics(filename_log, method_name, series_name, k, iters, exec_time, mse, variance, std_dev):
    file_exists = os.path.isfile(filename_log)
    with open(filename_log, mode='a', encoding='utf-8') as f:
        if not file_exists:
            f.write("Metodo,Serie,K,Iteraciones,Tiempo_s,MSE_Medio,Varianza,Std_Dev\n")
        f.write(f"{method_name},{series_name},{k},{iters},{exec_time:.6f},{mse:.6f},{variance:.6f},{std_dev:.6f}\n")

# --- GRÁFICAS ---
#Funcion graficar la serie
def draw(Y, breaking_points, filename, title="Regresión por Segmentos"):
    X = list(range(len(Y)))
    plt.plot(X, Y, color='blue', label='Serie')
    # Marcar puntos de corte con líneas verticales discontinuas
    if breaking_points:
        for bp in breaking_points:
            plt.axvline(x=bp, color='red', linestyle='--', linewidth=1, alpha=0.7, 
                       label='Puntos de corte' if bp == breaking_points[0] else None)

        # Dibujar las rectas de regresión por segmento
        for i in range(len(breaking_points)-1):
            start = breaking_points[i]
            end = breaking_points[i+1]

            # Crear X segmentada,e y segmentada
            X_seg = np.arange(start, end)
            y_seg = np.array(Y[start:end])

            A = np.vstack([X_seg, np.ones(len(X_seg))]).T

            rect, _, _, _ = np.linalg.lstsq(A, y_seg, rcond=None)
            m, c = rect

            # Predecimos con la ecuación de la recta y = mx + c
            y_pred = m * X_seg + c

            # Dibujar la recta de regresión
            plt.plot(X_seg, y_pred, color='orange', linewidth=2, label='Regresión' if i==0 else None)

    plt.title(filename)
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))  # Fuera a la derecha
    plt.savefig(filename[:-3] + 'png',dpi=300,bbox_inches='tight')

def draw_single_stat_with_variance(iteraciones, medias, desviaciones, titulo, algoritmo, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(iteraciones, medias, label='MSE Medio', color='blue', marker='o')
    plt.fill_between(iteraciones, np.array(medias) - np.array(desviaciones), 
                     np.array(medias) + np.array(desviaciones), 
                     color='red', alpha=0.2, label='Desviación Típica (±1σ)')
    plt.title(f"{titulo} (Precisión) - {algoritmo}\nArchivo: {filename}")
    plt.xlabel("Iteraciones")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def draw_variance_std_study(iteraciones, varianzas, desviaciones, nombre, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(iteraciones, varianzas, label='Varianza', color='purple', marker='s', linestyle='--')
    plt.plot(iteraciones, desviaciones, label='Desviación Típica', color='orange', marker='^')
    plt.title(f"Evolución de Estabilidad - {nombre}\nArchivo: {filename}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def analizar_distribucion_cuartiles(matriz_mse, nombre, filename):
    todos_los_datos = np.array([item for sublist in matriz_mse for item in sublist])
    mse_min, mse_max = np.min(todos_los_datos), np.max(todos_los_datos)
    bins = np.linspace(mse_min, mse_max, 5)
    counts, _ = np.histogram(todos_los_datos, bins=bins)
    
    etiquetas = ['Q1 (Mejores)', 'Q2', 'Q3', 'Q4 (Peores)']
    plt.figure(figsize=(10, 6))
    bars = plt.bar(etiquetas, counts, color=['#2ca02c', '#94df94', '#ffcc00', '#d62728'], edgecolor='black')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, int(bar.get_height()), ha='center')

    plt.title(f"Frecuencia por Rango de Calidad\nRango: [{mse_min:.4f} - {mse_max:.4f}]\nAlgoritmo: {nombre}")
    plt.ylabel("Nº Soluciones")
    plt.show()

def draw_convergence_best(history, nombre):
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='green', linewidth=2, label='Mejor MSE encontrado')
    plt.title(f"Convergencia hacia el Óptimo - {nombre}")
    plt.xlabel("Iteraciones Totales")
    plt.ylabel("MSE")
    plt.yscale('log') # Escala logarítmica para ver mejor las mejoras pequeñas
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()
