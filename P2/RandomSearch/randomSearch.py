import random
import sys
import os
import time
import psutil
import threading
import statistics
import csv

# Añadimos las rutas absolutas para evitar problemas de importación
ruta_padre = os.path.abspath("..")
ruta_genetico = os.path.abspath("../Algoritmo_Genetico")
if ruta_padre not in sys.path: sys.path.append(ruta_padre)
if ruta_genetico not in sys.path: sys.path.append(ruta_genetico)

from Algoritmo_Genetico.individuals import Individual
from Algoritmo_Genetico.model import evaluate_solution

# Búsqueda aleatoria.
def randomSearch(ind: Individual, n_iter):
    best_score = 0
    best_ind = None
    all_scores = [] # Aquí guardaremos todas las notas para la varianza y la media
    
    print(f"Total iteraciones programadas: {n_iter} (Pulsa Ctrl+C para detener y guardar)")
    print("-" * 50)

    # Envolvemos el bucle en un try/except para capturar el Ctrl+C
    try:
        for i in range(0, n_iter):
            ind.n_estimators = random.randint(10, 300)
            ind.max_depth = random.randint(2, 30)
            ind.min_samples_split = random.randint(2, 20)
            ind.min_samples_leaf = random.randint(1, 20)
            ind.max_features = random.uniform(0.1, 1)
            ind.bootstrap = random.choice([True, False])
            ind.criterion = random.choice([0, 1])    
            ind.class_weight = random.choice([0, 1]) 
            ind.max_leaf_nodes = random.randint(10, 200)
            ind.min_impurity_decrease = random.uniform(0, 0.1)
            ind.random_state = random.randint(0, 1000)
            
            try:
                ind.score = evaluate_solution(ind)
                all_scores.append(ind.score)
                
                print(f"[{i + 1}/{n_iter}] -> Score: {ind.score:.4f}")
                
                if best_score < ind.score:
                    best_score = ind.score
                    # Instanciamos uno nuevo para no perder los parámetros del mejor
                    best_ind = Individual(ind.n_estimators, ind.max_depth, ind.min_samples_split, 
                                          ind.min_samples_leaf, ind.max_features, ind.bootstrap, 
                                          ind.criterion, ind.class_weight, ind.max_leaf_nodes, 
                                          ind.min_impurity_decrease, ind.random_state, 0)
                    best_ind.score = best_score
            except Exception as e:
                print(f"[{i + 1}/{n_iter}] -> Error evaluando: {e}")
                
    except KeyboardInterrupt:
        # Esto se ejecuta mágicamente cuando pulsas Ctrl+C
        print(f"\n[!] Ejecución interrumpida por el usuario con Ctrl+C (Iteración {i}).")
        print("[!] Guardando progreso y calculando estadísticas parciales...")

    return best_ind, all_scores

if __name__ == "__main__":
    
    # 1. INICIALIZAMOS EL MONITOR DE RECURSOS
    keep_monitoring = True
    cpu_usage = []
    mem_usage = []

    def monitor():
        p = psutil.Process()
        while keep_monitoring:
            try:
                cpu_usage.append(p.cpu_percent(interval=0.1))
                mem_usage.append(p.memory_info().rss / (1024 * 1024)) # MB
            except:
                break

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    
    # 2. INICIAMOS EL CRONÓMETRO Y EJECUTAMOS
    start_time = time.time()
    
    # Individuo base que el algoritmo machacará con parámetros aleatorios
    ind_base = Individual(100, 10, 2, 1, 0.5, True, 0, 0, 100, 0.0, 42, 0)
    
    # Iteraciones máximas solicitadas (17.496 para igualar al Grid Search)
    num_combinaciones = 17496 
    
    best_ind, scores_list = randomSearch(ind_base, n_iter=num_combinaciones)
    
    end_time = time.time()
    
    # 3. APAGAMOS EL MONITOR
    keep_monitoring = False
    monitor_thread.join()

    # Calculamos cuántas iteraciones logró hacer realmente antes del Ctrl+C
    combinaciones_reales = len(scores_list)

    # 4. CÁLCULO DE TODAS LAS ESTADÍSTICAS SOLICITADAS
    tiempo_total = end_time - start_time
    varianza = statistics.variance(scores_list) if combinaciones_reales > 1 else 0
    desviacion = statistics.stdev(scores_list) if combinaciones_reales > 1 else 0
    media_accuracy = statistics.mean(scores_list) if combinaciones_reales > 0 else 0
    max_accuracy = max(scores_list) if combinaciones_reales > 0 else 0
    min_accuracy = min(scores_list) if combinaciones_reales > 0 else 0
    cpu_media = statistics.mean(cpu_usage) if cpu_usage else 0
    memoria_mb = max(mem_usage) if mem_usage else 0

    # 5. GUARDADO EN CSV
    archivo_csv = "../resultados_estadisticas.csv" 
    existe = os.path.isfile(archivo_csv)
    
    # Solo guardamos en el CSV si ha logrado completar al menos 1 iteración
    if combinaciones_reales > 0:
        with open(archivo_csv, "a", newline="") as f:
            writer = csv.writer(f)
            
            # Si el archivo no existe, escribimos las cabeceras primero
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
                
            # Escribimos los datos de esta ejecución
            writer.writerow([
                "Random Search", 
                round(varianza, 6), 
                round(desviacion, 6), 
                round(tiempo_total, 2), 
                combinaciones_reales, # <--- Se guarda el total real
                round(media_accuracy, 4), 
                round(max_accuracy, 4), 
                round(min_accuracy, 4), 
                round(cpu_media, 2), 
                round(memoria_mb, 2)
            ])

        print("\n" + "=" * 50)
        print(f"BEST SCORE OBTENIDO: {max_accuracy:.4f}")
        if best_ind: best_ind.features()
        print(f"[+] Estadísticas exportadas correctamente a '{archivo_csv}'")
    else:
        print("\n[!] Se interrumpió el programa antes de terminar ninguna iteración. No se guardan datos en el CSV.")