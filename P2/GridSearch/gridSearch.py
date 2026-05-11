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

def grid_search():
    param_grid = {
        "n_estimators": [10, 155, 300],
        "max_depth": [2, 15, 30],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 10, 20],
        "max_features": [0.1, 0.5, 1],
        "bootstrap": [True, False],
        "criterion": ["entropy", "gini"],
        "class_weight": ["balanced", None],
        "max_leaf_nodes": [10, 100, 200],
        "min_impurity_decrease": [0, 0.05, 0.1],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    results = []
    all_scores = []
    total = len(combinations)

    print(f"Total combinations: {total}")
    print("-" * 50)

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"[{idx + 1}/{total}] {params}")

        try:
            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            mean_score = scores.mean()

            results.append({**params, "score": mean_score})
            all_scores.append(mean_score)
            print(f"  -> Score: {mean_score:.4f}")
        except Exception as e:
            print(f"  -> Error: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 50)
    print("TOP 5 CONFIGURATIONS:")
    for i, res in enumerate(results[:5]):
        print(f"{i + 1}. Score: {res['score']:.4f}")
        params_only = {k: v for k, v in res.items() if k != "score"}
        print(f"   {params_only}")

    print("\n" + "=" * 50)
    print(f"BEST: {results[0]['score']:.4f}")

    best_params_only = {k: v for k, v in results[0].items() if k != "score"}
    print(best_params_only)

    return best_params_only, all_scores, total


if __name__ == "__main__":
    
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
    
    best_params, scores_list, num_combinaciones = grid_search()
    
    end_time = time.time()
    
    keep_monitoring = False
    monitor_thread.join()

    tiempo_total = end_time - start_time
    varianza = statistics.variance(scores_list) if len(scores_list) > 1 else 0
    desviacion = statistics.stdev(scores_list) if len(scores_list) > 1 else 0
    media_accuracy = statistics.mean(scores_list) if scores_list else 0
    max_accuracy = max(scores_list) if scores_list else 0
    min_accuracy = min(scores_list) if scores_list else 0
    cpu_media = statistics.mean(cpu_usage) if cpu_usage else 0
    memoria_mb = max(mem_usage) if mem_usage else 0  # El pico de memoria máxima usada

    archivo_csv = "resultados_estadisticas.csv"
    existe = os.path.isfile(archivo_csv)
    
    with open(archivo_csv, "a", newline="") as f:
        writer = csv.writer(f)
        
        if not existe:
            writer.writerow([
                "Algoritmo", 
                "Varianza", 
                "Desviacion_Tipica", 
                "Tiempo_Segundos", 
                "Num_Combinaciones", 
                "Media_Accuracy", 
                "Max_Accuracy", 
                "Min_Accuracy", 
                "CPU_Media_%", 
                "Memoria_MB"
            ])
            
        writer.writerow([
            "Grid Search", 
            round(varianza, 6), 
            round(desviacion, 6), 
            round(tiempo_total, 2), 
            num_combinaciones, 
            round(media_accuracy, 4), 
            round(max_accuracy, 4), 
            round(min_accuracy, 4), 
            round(cpu_media, 2), 
            round(memoria_mb, 2)
        ])

    print("\n=======================================================")
    print(f"[+] Estadísticas exportadas correctamente a '{archivo_csv}'")