import time
import csv
import os
import metrics as me
from BusquedaAleatoria import serialRandomSearch, paralelRandomSearch

def save_data(csv_file, algorithm, series_filename, max_iter, execution, MSE, time_elapsed):
    os.makedirs(os.path.dirname(csv_file) if os.path.dirname(csv_file) else ".", exist_ok=True)
    filexist = os.path.isfile(csv_file)
    
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not filexist:
            writer.writerow(["Algorithm", "Series", "Iterations", "Execution", "MSE", "Time"])
        writer.writerow([algorithm, series_filename, max_iter, execution, MSE, time_elapsed])

if __name__ == "__main__":
    series_list = [
        ("TS1.txt", 9),
        ("TS2.txt", 15),
        ("TS3.txt", 25),
        ("TS4.txt", 50)
    ]

    algorithms = [
        (serialRandomSearch, "RS_Serie"),
        (paralelRandomSearch, "RS_Paralelo")
    ]

    iter_values = [50, 100, 150, 200]
    repeticiones = 20
    csv_path = "./test_files/RS_results.csv"

    total_ejecuciones = len(series_list) * len(algorithms) * len(iter_values) * repeticiones
    print(f"=========================================================")
    print(f" INICIANDO ESTUDIO AUTOMATIZADO DE RANDOM SEARCH (RS)    ")
    print(f"=========================================================")
    print(f"Total a ejecutar: {total_ejecuciones} pruebas.\n")

    contador = 0
    for filename, k_segments in series_list:
        series_data = me.readSeries(filename)
        print(f"\n>>> Procesando serie: {filename} (k={k_segments}) <<<")

        for func, alg_name in algorithms:
            print(f"\n--- Evaluando {alg_name} ---")

            for max_iter in iter_values:
                mejor_mse_global = float('inf')
                mejores_ptos_global = []

                print(f"  -> Con {max_iter} iteraciones:")

                for i in range(repeticiones):
                    contador += 1
                    start_time = time.time()
                    
                    ptos = func(series_data, k_segments, max_iter)
                    
                    time_elapsed = time.time() - start_time
                    mse = me.avgMSE(series_data, ptos)
                    
                    save_data(csv_path, alg_name, filename, max_iter, (i+1), mse, time_elapsed)

                    if mse < mejor_mse_global:
                        mejor_mse_global = mse
                        mejores_ptos_global = list(ptos)

                    print(f"    [{contador:04d}/{total_ejecuciones}] Rep {i+1:02d} | MSE: {mse:.4f} | Tiempo: {time_elapsed:.4f}s")
                
                print(f"    => Mejor resultado de {alg_name} ({max_iter} iter) en {filename}! MSE: {mejor_mse_global:.4f}")
                
                titulo_grafica = f"Mejor {alg_name} ({max_iter} iter) - {filename} (MSE: {mejor_mse_global:.4f})"
                me.draw(series_data, mejores_ptos_global, filename, title=titulo_grafica)

    print(f"\n=========================================================")
    print(f" ¡ESTUDIO FINALIZADO! Los datos están en {csv_path}")
    print(f"=========================================================")
