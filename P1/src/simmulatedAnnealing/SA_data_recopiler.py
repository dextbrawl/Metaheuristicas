import csv
import os
import simmulatedAnnealing as sa
import metrics as me
import itertools

# Podría añadirse el número de la iteración en la que realmente ha parado cada uno por sus criterios de parada independientes
def save_data(csv_file,algorithm, series_filename, max_iter, initialTemperature, finalTemperature, L, coolingFunction, execution, MSE, time):
    filexist = os.path.isfile(csv_file)
    with open("./test_files/SA_results.csv", mode= "a", newline= "") as f:
        writer = csv.writer(f)

        if not filexist:
            writer.writerow(["Algorithm", "Series", "Iterations", "T0", "Tf", "L", "coolingFunction","Execution", "MSE", "Time"])

        writer.writerow([algorithm, series_filename, max_iter, initialTemperature, finalTemperature, L, coolingFunction, execution, MSE, time])

if __name__ == "__main__":

    T0_values = [200, 50, 10]
    Tf_values = [10, 1, 0.01]
    L_values = [10, 30, 50]

    coolingFunctions = [ (sa.linealCooling, "Linear"), (sa.logarithmCooling, "Logarithm"), (sa.geometricCooling, "Geometric")]

    combinations = list(itertools.product(T0_values, Tf_values, L_values, coolingFunctions))

    filename = "TS1.txt"            
    k_segments = 9                  
                                
    series = me.readSeries(filename)
    c = 0
    for T0, Tf, L, (coolingFunction, function_name) in combinations:
        for i in range(20):
            c += 1
            max_iter = 200

            algorithm_name = f"SA_{function_name}Cooling"

            breaking_points, mse, time_elapsed = sa.simmulatedAnnealing(series, k_segments, T0, L, Tf, coolingFunction, max_iter)

            print(f" línea: {c} Config: T0={T0:<3d} | Tf={Tf:<5.2f} | L={L:<2d} | Cool={function_name:<9s} --> Rep {i+1:02d} | Tiempo: {time_elapsed:.2f}s")

            save_data("SA_results", algorithm_name, filename, max_iter, T0, Tf, L, function_name, (i+1), mse, time_elapsed)

