import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import simmulatedAnnealing as sa

# Podría añadirse el número de la iteración en la que realmente ha parado cada uno por sus criterios de parada independientes
def save_data(csv_file,algorithm, series_filename, max_iter, initialTemperature, finalTemperature, L, coolingFunction,execution, MSE, time):
    filexist = os.path.isfile(csv_file)
    with open("./data/SA_data", mode= "a", newline= "") as f:
        writer = csv.writer(f)

        if not filexist:
            writer.writerow(["Algorithm", "Series", "Iterations", "T0", "Tf", "L", "coolingFunction","Execution", "MSE", "Time"])

        writer.writerow([algorithm, series_filename, max_iter, execution, MSE, time])

if __name__ == "__main__":
    for i in range(20):
        filename, k_segments = me.select_series()

        series = me.readSeries(filename)
    
        max_iter = 200

        sa.simmulatedAnnealing(series, k_segments, 50, 30, 0.01, sa.linealCooling, max_iter)
