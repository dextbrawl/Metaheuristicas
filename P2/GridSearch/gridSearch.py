from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time
import psutil
import threading
import statistics
import csv
import os

data = pd.read_csv("../winequality-red.csv", sep=";")
data["quality"] = (data["quality"] >= 6).astype(int)
X = data.drop("quality", axis=1)
y = data["quality"]

def grid_search(n_iters):
    param_grid = {
        "n_estimators": [50, 100, 150, 200, 300], # 5 opciones
        "max_depth": [5, 10, 20, 30, None],       # 5 opciones
        "min_samples_split": [2, 5, 10, 20],      # 4 opciones
        "min_samples_leaf": [1, 2, 4, 8, 10],     # 5 opciones
        "max_features": ["sqrt", "log2"],         # 2 opciones (Total = 1000 combinaciones)
        "bootstrap": [True],                      
        "criterion": ["gini"],                    
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))[:n_iters] # Cortamos exactamente al numero de iteraciones

    results = []
    all_scores = []
    total = len(combinations)

    print(f"\nTotal combinations to evaluate: {total}")
    print("-" * 50)

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        try:
            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            mean_score = scores.mean()

            results.append({**params, "score": mean_score})
            all_scores.append(mean_score)
        except Exception as e:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[0], all_scores

if __name__ == "__main__":
    lista_iteraciones = [100, 200, 300, 400, 500]
    
    for iters in lista_iteraciones:
        print(f"\n===============================================")
        print(f"🚀 Ejecutando GRID SEARCH - {iters} Iteraciones")
        print(f"===============================================")
        
        keep_monitoring = True
        cpu_usage = []
        mem_usage = []

        def monitor():
            p = psutil.Process()
            while keep_monitoring:
                try:
                    cpu_usage.append(p.cpu_percent(interval=0.1))
                    mem_usage.append(p.memory_info().rss / (1024 * 1024))
                except:
                    break

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        
        start_time = time.time()
        
        best, all_scores = grid_search(iters)
        
        end_time = time.time()
        keep_monitoring = False
        monitor_thread.join()

        tiempo_total = end_time - start_time
        mejor_resultado = best["score"]
        varianza = statistics.variance(all_scores) if len(all_scores) > 1 else 0
        desviacion = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
        cpu_media = statistics.mean(cpu_usage) if cpu_usage else 0
        memoria_pico = max(mem_usage) if mem_usage else 0

        archivo_csv = "../estadisticas_algoritmos.csv"
        existe = os.path.isfile(archivo_csv)
        
        with open(archivo_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not existe:
                writer.writerow(["Algoritmo", "Iteraciones", "Mejor_Accuracy", "Varianza", "Desviacion", "Tiempo_Segundos", "CPU_Media_%", "Memoria_Pico_MB"])
            writer.writerow(["Grid Search", iters, round(mejor_resultado, 4), round(varianza, 6), round(desviacion, 6), round(tiempo_total, 2), round(cpu_media, 2), round(memoria_pico, 2)])

        print(f"[+] Guardado: Grid Search | {iters} iters | Acc: {mejor_resultado:.4f} | Tiempo: {tiempo_total:.2f}s")