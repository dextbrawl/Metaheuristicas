import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Podría añadirse el número de la iteración en la que realmente ha parado cada uno por sus criterios de parada independientes
def save_data(csv_file,algorithm, series_filename, max_iter, execution, MSE, time):
    filexist = os.path.isfile(csv_file)
    with open("csv_file", mode= "a", newline= "") as f:
        writer = csv.writer(f)

        if not filexist:
            writer.writerow(["Algorithm", "Series", "Iterations", "Execution", "MSE", "Time"])

        writer.writerow([algorithm, series_filename, max_iter, execution, MSE, time])

