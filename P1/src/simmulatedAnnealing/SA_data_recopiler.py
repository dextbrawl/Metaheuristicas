import csv
import os
import itertools
import simmulatedAnnealing as sa
import sys
aux_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'aux'))
sys.path.append(aux_path)
import metrics as me

"""Almacena los datos pasados como argumento en un fichero csv"""
def save_data(csv_file, algorithm, series_filename, max_iter, initialTemperature, finalTemperature, L, coolingFunction, execution, MSE, time_elapsed):
    os.makedirs(os.path.dirname(csv_file) if os.path.dirname(csv_file) else ".", exist_ok=True)
    filexist = os.path.isfile(csv_file)
    
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not filexist:
            writer.writerow(["Algorithm", "Series", "Iterations", "T0", "Tf", "L", "coolingFunction", "Execution", "MSE", "Time"])

        writer.writerow([algorithm, series_filename, max_iter, initialTemperature, finalTemperature, L, coolingFunction, execution, MSE, time_elapsed])

if __name__ == "__main__":
    
    # Valores de prueba que usaremos para el estudio de SA
    T0_values = [50.0, 10.0, 1.0]
    Tf_values = [1.0, 0.1, 0.001]
    L_factors = [0.5, 2.0, 5.0] 

    coolingFunctions = [(sa.geometricCooling, "Geometric")]

    # Genera todas las combinaciones posibles
    combinations = list(itertools.product(T0_values, Tf_values, L_factors, coolingFunctions))

    series_list = [
        ("TS1.txt", 9),
        ("TS2.txt", 15), 
        ("TS3.txt", 25), 
        ("TS4.txt", 50)  
    ]

    csv_path = "./test_files/SA_hiperparametros.csv"
    repeticiones = 20

    contador = 0
    for filename, k_segments in series_list:
        series = me.readSeries(filename)
        vecindario = 2 * k_segments
        
        print(f"\n--- Analizando {filename} (Vecindario: {vecindario}) ---")

        for T0, Tf, factor_L, (coolingFunction, function_name) in combinations:
            L_real = int(factor_L * vecindario) # En vez de hacerlo arbitrario, el tamaño L se calcula con un factor del vecindario

            for i in range(repeticiones):
                contador += 1
                max_iter = 200
                algorithm_name = f"SA_{function_name}Cooling"

                breaking_points, mse, time_elapsed = sa.simmulatedAnnealing(
                    series, k_segments, T0, L_real, Tf, coolingFunction, max_iter
                )

                save_data(csv_path, algorithm_name, filename, max_iter, T0, Tf, L_real, function_name, (i+1), mse, time_elapsed)

                print(f"Config: T0={T0:<4.1f} | Tf={Tf:<5.3f} | L_factor={factor_L} (L_real={L_real}) --> MSE: {mse:.4f}")
