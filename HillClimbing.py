import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Funciones auxiliares

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

def getRandBreakingPoints(n_points, k_segments): # Función para generar puntos de corte aleatorios
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

def print_segments(breaking_points):
    for i in range(len(breaking_points) - 1):
        print(f'[{breaking_points[i]}][{breaking_points[i+1]}]')

def hillClimbingSearch(series, k_segments, prev_breaking_points): # Sacar nuevos puntos vecinos que sean mejores que los anteriores
    improved = True        
    while improved:
        i = 1
        improved = False
        while i < k_segments :
            prev_mean_mse = avgMSE(series,prev_breaking_points)
            clear_screen()
            print('-- HILL CLIMBING SEARCH --')
            print_segments(prev_breaking_points)
            print(f'Average MSE: {avgMSE(series,prev_breaking_points)}')
            inc_breaking_points = prev_breaking_points.copy()
            if prev_breaking_points[i] + 1 < prev_breaking_points[i+1]:
                inc_breaking_points[i] += 1
            dec_breaking_points = prev_breaking_points.copy()
            if prev_breaking_points[i] - 1 > prev_breaking_points[i-1]:
                dec_breaking_points[i] -= 1
            mean_mse_inc = avgMSE(series,inc_breaking_points)
            mean_mse_dec = avgMSE(series,dec_breaking_points)

            if mean_mse_inc < prev_mean_mse and mean_mse_dec >= prev_mean_mse: # Mejora incrementando y empeora decrementando
                prev_breaking_points = inc_breaking_points
                improved = True

            elif mean_mse_inc >= prev_mean_mse and mean_mse_dec < prev_mean_mse: # Mejora decrementando y empeora incrementando
                prev_breaking_points = dec_breaking_points
                improved = True

            elif mean_mse_inc < prev_mean_mse and mean_mse_dec < prev_mean_mse: # Mejora en los dos casos
                if mean_mse_inc < mean_mse_dec: #Mejora más incrementando
                    prev_breaking_points = inc_breaking_points
                    improved = True
                else: # Mejora más (o igual) decrementando
                    prev_breaking_points = dec_breaking_points
                    improved = True
            else: # Empeora en ambos casos
                i += 1
    return prev_breaking_points

series = readSeries('TS1.txt')
k_segments = 9
prev_breaking_points = getRandBreakingPoints(len(series),k_segments)
best_breaking_points = hillClimbingSearch(series,k_segments,prev_breaking_points)
                
