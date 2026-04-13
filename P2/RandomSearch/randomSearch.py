import random
import sys
import os 
import time
import psutil
import threading
import statistics
import csv

ruta_padre = os.path.abspath("..")
ruta_genetico = os.path.abspath("../Algoritmo_Genetico")
sys.path.append(ruta_padre)
sys.path.append(ruta_genetico) 

from Algoritmo_Genetico.individuals import Individual
from Algoritmo_Genetico.model import evaluate_solution

def randomSearch(ind: Individual, n_iter):
    best_score = 0
    best_ind = None
    all_scores = []
    
    for _ in range(0, n_iter):
        ind.n_estimators = random.randint(10, 500)
        ind.max_depth = random.randint(1, 50)
        ind.min_samples_split = random.randint(2, 20)
        ind.min_samples_leaf = random.randint(1, 10)
        ind.max_features = random.choice(["sqrt", "log2", None])
        ind.bootstrap = random.choice([True, False])
        ind.criterion = random.choice([0, 1]) 
        ind.class_weight = random.choice([0, 1]) 
        ind.max_leaf_nodes = random.randint(10, 500)
        ind.min_impurity_decrease = random.uniform(0, 0.1)
        ind.random_state = random.randint(0, 1000)
        
        ind.score = evaluate_solution(ind)
        all_scores.append(ind.score)
        
        if best_score < ind.score:
            best_score = ind.score
            best_ind = Individual(ind.n_estimators, ind.max_depth, ind.min_samples_split, 
                                  ind.min_samples_leaf, ind.max_features, ind.bootstrap, 
                                  ind.criterion, ind.class_weight, ind.max_leaf_nodes, 
                                  ind.min_impurity_decrease, ind.random_state, 0)
            best_ind.score = best_score
            
    return best_ind, all_scores

if __name__ == "__main__":
    lista_iteraciones = [100, 200, 300, 400, 500]
    
    for iters in lista_iteraciones:
        print(f"\n===============================================")
        print(f"🚀 Ejecutando RANDOM SEARCH - {iters} Iteraciones")
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
        
        ind_base = Individual(100, 10, 2, 1, "sqrt", True, 0, 0, 100, 0.0, 42, 0)
        best, all_scores = randomSearch(ind_base, n_iter=iters)
        
        end_time = time.time()
        keep_monitoring = False
        monitor_thread.join()

        tiempo_total = end_time - start_time
        mejor_resultado = best.score
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
            writer.writerow(["Random Search", iters, round(mejor_resultado, 4), round(varianza, 6), round(desviacion, 6), round(tiempo_total, 2), round(cpu_media, 2), round(memoria_pico, 2)])

        print(f"[+] Guardado: Random Search | {iters} iters | Acc: {mejor_resultado:.4f} | Tiempo: {tiempo_total:.2f}s")