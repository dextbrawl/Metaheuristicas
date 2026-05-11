import random
import numpy as np
import time
import psutil
import threading
import statistics
import csv
import os
import sys

# Ajuste de rutas por si se ejecuta desde otra carpeta
ruta_padre = os.path.abspath("..")
if ruta_padre not in sys.path: sys.path.append(ruta_padre)

import CreatePopulation
import crossing
import individuals
import replacement
import Selection
import model

mutation_prob = 0.5
cross_prob = 0.5

MAX_MUTATION_PROB, MIN_MUTATION_PROB = 0.8, 0.1
MAX_CROSSING_PROB, MIN_CROSSING_PROB = 0.9, 0.2
STAGNATION_THRESHOLD = 0.12

def populationVariance(population):
    vectors = np.array([CreatePopulation.NormaliseIndividual(ind) for ind in population])
    return np.mean(np.var(vectors, axis=0))

def changeProbabilities(mutation_prob, cross_prob, population):
    scores = [ind.score for ind in population]
    mean = np.mean(scores)
    variance = populationVariance(population)
    stagnated = False

    if variance < STAGNATION_THRESHOLD:
        mutation_prob = min(mutation_prob + 0.1, MAX_MUTATION_PROB)
        cross_prob = max(cross_prob - 0.1, MIN_CROSSING_PROB)
        stagnated = True
    else:   
        mutation_prob = max(mutation_prob - 0.1, MIN_MUTATION_PROB)
        cross_prob = min(cross_prob + 0.1, MAX_CROSSING_PROB)
        stagnated = False
    
    with open("prob_data.csv", "a") as f:
        f.write(f"\n{mean},{variance},{mutation_prob},{cross_prob}")
    
    return mutation_prob, cross_prob, stagnated

if __name__ == "__main__":
    
    # Hemos arreglado las comillas anidadas que había en el string original
    print(f"Prueba de modelo base: {model.evaluate_solution(individuals.Individual(250, 43, 15, 35, 'sqrt', False, 1, 0, 155, 0, 42, 0)):.4f}")
    
    # 1. INICIALIZAMOS EL MONITOR DE RECURSOS
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

    # 2. INICIAMOS EL CRONÓMETRO GLOBAL
    start_time = time.time()
    
    # ========================================================
    # VARIABLES SOLICITADAS
    # ========================================================
    population_size = 50
    max_iter = 349
    
    print(f"\nCreando la población inicial de {population_size} individuos (Esto puede tardar un poco)...")
    population = CreatePopulation.CreateSequentialPopulation(population_size, 0.5, 150)
    print("Población inicial creada con éxito.")
    
    stagnated = False
    
    # Variable para contar exactamente cuántas configuraciones se evalúan (Aproximación precisa)
    num_combinaciones = population_size 
    
    with open("prob_data.csv", "w") as f:
        f.write("media_score,varianza_params,prob_mut,prob_cross")
    
    # Preparamos el CSV Principal (El mismo de Grid y Random)
    archivo_csv = "../resultados_estadisticas.csv"
    existe = os.path.isfile(archivo_csv)
    
    if not existe:
        with open(archivo_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Algoritmo", "Varianza", "Desviacion_Tipica", "Tiempo_Segundos", 
                "Num_Combinaciones", "Media_Accuracy", "Max_Accuracy", "Min_Accuracy", 
                "CPU_Media_%", "Memoria_MB"
            ])

    print(f"\nProbabilidad de mutación y cruce iniciales: {mutation_prob}, {cross_prob}")
    print("-" * 50)
    print("Pulsa Ctrl+C en cualquier momento para detener la evolución y guardar lo conseguido.")
    
    i = 0
    try:
        # Bucle de Generaciones
        while i < max_iter:
            selected = Selection.TournamentSelection(population)
            children = []
            
            for j in range(0, len(selected)-1, 2):
                n_points = 4
                c_prob = random.random()
            
                if c_prob < cross_prob:
                    if random.random() < 0.5:
                        child1 = crossing.n_point_crossing(selected[j], selected[j + 1], n_points)
                        child2 = crossing.n_point_crossing(selected[j], selected[j + 1], n_points)
                    else:
                        child1 = crossing.uniform_crossing(selected[j], selected[j + 1])
                        child2 = crossing.uniform_crossing(selected[j], selected[j + 1])

                    children.append(child1)
                    children.append(child2)
                    num_combinaciones += 2 # Cada hijo creado es un modelo nuevo evaluado
                else:
                    children.append(selected[j])
                    children.append(selected[j + 1])

            for j in range(0, len(children)):
                m_prob = random.uniform(0.0, 1.0)
                if m_prob < mutation_prob:
                    children[j].mutate()
                    num_combinaciones += 1 # Al mutar, se vuelve a evaluar

            if stagnated:
                population = replacement.replaceWorst(population, 5)
                num_combinaciones += 5 # Los individuos aleatorios creados aquí también se evalúan
            else:
                population = replacement.replaceWorstWithChildren(population, children)
            
            mutation_prob, cross_prob, stagnated = changeProbabilities(mutation_prob, cross_prob, population)
            
            # ==========================================================
            # CÁLCULO DE ESTADÍSTICAS DE LA GENERACIÓN ACTUAL
            # ==========================================================
            current_scores = [ind.score for ind in population]
            
            tiempo_actual = time.time() - start_time
            varianza = statistics.variance(current_scores) if len(current_scores) > 1 else 0
            desviacion = statistics.stdev(current_scores) if len(current_scores) > 1 else 0
            media_accuracy = statistics.mean(current_scores)
            max_accuracy = max(current_scores)
            min_accuracy = min(current_scores)
            cpu_media = statistics.mean(cpu_usage) if cpu_usage else 0
            memoria_mb = max(mem_usage) if mem_usage else 0
            
            # Formateamos el nombre para que en Excel puedas hacer gráficos por generación
            nombre_algoritmo = f"Genetico - Gen {i+1}"
            
            with open(archivo_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    nombre_algoritmo, 
                    round(varianza, 6), 
                    round(desviacion, 6), 
                    round(tiempo_actual, 2), 
                    num_combinaciones, 
                    round(media_accuracy, 4), 
                    round(max_accuracy, 4), 
                    round(min_accuracy, 4), 
                    round(cpu_media, 2), 
                    round(memoria_mb, 2)
                ])

            print(f"[Gen {i+1}/{max_iter}] Max Acc: {max_accuracy:.4f} | Media: {media_accuracy:.4f} | Tiempo: {tiempo_actual:.2f}s | Evals: {num_combinaciones}")
            i += 1
            
    except KeyboardInterrupt:
        print(f"\n[!] Ejecución interrumpida por el usuario en la generación {i+1}.")
        print("[!] Guardando estado final...")

    # ==========================================================
    # APAGAMOS EL MONITOR Y BUSCAMOS AL GANADOR GLOBAL
    # ==========================================================
    keep_monitoring = False
    monitor_thread.join()

    max_score = 0
    final_individual = None
    for ind in population:
        if(max_score < ind.score):
            max_score = ind.score
            final_individual = ind
            
    print("\n" + "=" * 50)
    print(f"Mejor Score Final Obtenido: {max_score:.4f}")
    if final_individual:
        final_individual.features()
        
    print(f"[+] Progreso del Algoritmo Genético guardado en '{archivo_csv}'")