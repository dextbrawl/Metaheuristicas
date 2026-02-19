import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt #Para pintar las curvas

# Funciones auxiliares
def draw(filename,breaking_points):
    #Codigo de draw
    Y=readSeries(filename)
    X=list(range(len(Y)))
    plt.plot(X,Y,color='blue',label='Serie')

    #Marcar segmentos
    if breaking_points:
        X_cuts = breaking_points
        Y_cuts = [Y[i] for i in breaking_points]
        plt.scatter(X_cuts, Y_cuts, color='red', s=50, label='Puntos de corte', zorder=5)

    plt.title(filename)
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.grid(True)
    plt.legend()
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

def readSeries(filename) -> list: # Función para crear la lista que define la serie
    data = np.loadtxt(filename).tolist()
    return data

def getBreakingPoints(n_points, k_segments): # Función para generar puntos de corte aleatorios
    breaking_points = [0,n_points-1]

    for _ in range(k_segments -1):
        n = random.randint(0,n_points-1)
        while n in breaking_points:
            n = random.randint(0,n_points-1)
        breaking_points.append(n)
    breaking_points.sort()

    return breaking_points

def clear_screen():
    print('\033[2J\033[H', end='')

def randomSearch(series: list, k_segments):
    clear_screen()
    print(" -- RANDOM SEARCH --")
    size = len(series)
    
    breaking_points = getBreakingPoints(size,k_segments)
    avg_mse = avgMSE(series,breaking_points)
    
    print("SEGMENTS")
    for i in range(k_segments-1):
        print("     [",breaking_points[i],",",breaking_points[i+1],"]")
    print("Average MSE: ", avg_mse)

    for _ in range(10):
        new_breaking_points = getBreakingPoints(size, k_segments)
        new_avg_mse = avgMSE(series,new_breaking_points)

        if new_avg_mse < avg_mse :
            breaking_points = new_breaking_points
            avg_mse = new_avg_mse
            clear_screen()
            print(" -- RANDOM SEARCH --")
           
            print("SEGMENTS")
            for i in range(k_segments-1):
                print("     [",breaking_points[i],",",breaking_points[i+1],"]")
            print("Average MSE: ", avg_mse)
    #Hago return de los puntos para usarlos en draw
    return breaking_points

best_breaking_points =randomSearch(readSeries("TS1.txt"),9)
draw("TS1.txt",best_breaking_points)
