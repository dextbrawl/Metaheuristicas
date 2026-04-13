import random
import numpy as np
import time
import psutil
import threading
import statistics
import csv
import os

import CreatePopulation
import crossing
import individuals
import replacement
import Selection
import model

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
    lista_iteraciones = [100, 200, 300, 400, 500]
    
    for iters_totales in lista_iteraciones:
        print(f"\n=====================================================")
        print(f"🚀 Ejecutando ALGORITMO GENÉTICO - {iters_totales} Evaluaciones")
        print(f"=====================================================")

        keep_monitoring = True
        cpu_usage = []
        mem_usage = []

        def monitor():
            p = psutil.Process()
            while keep_monitoring:
                try:
                    cpu_usage.append(p.cpu_percent(interval=0.1))
                    mem_usage.append(p.memory_info().rss / (1024 * 1024))
                except psutil.NoSuchProcess:
                    break

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        start_time = time.time()
        
        # Reiniciamos variables por cada prueba
        mutation_prob = 0.5
        cross_prob = 0.5
        stagnated = False
        all_scores = []
        
        with open("prob_data.csv", "w") as f:
            f.write("media_score,varianza_params,prob_mut,prob_cross")

        # Calculo de generaciones para clavar las iteraciones
        population_size = 20
        max_generations = (iters_totales - population_size) // population_size
        
        population = CreatePopulation.CreateSequentialPopulation(population_size, 0.5, 150)
        
        i = 0
        while i < max_generations:
            all_scores.extend([ind.score for ind in population])
            
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
                else:
                    children.append(selected[j])
                    children.append(selected[j + 1])

            for j in range(0, len(children)):
                m_prob = random.uniform(0.0, 1.0)
                if m_prob < mutation_prob:
                    children[j].mutate()

            if stagnated:
                population = replacement.replaceWorst(population, 5)
            else:
                population = replacement.replaceWorstWithChildren(population, children)
            
            i += 1
            mutation_prob, cross_prob, stagnated = changeProbabilities(mutation_prob, cross_prob, population)

        max_score = 0
        for j in population:
            if max_score < j.score:
                max_score = j.score
                
        end_time = time.time()
        keep_monitoring = False
        monitor_thread.join()

        tiempo_total = end_time - start_time
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
            writer.writerow(["Algoritmo Genetico", iters_totales, round(max_score, 4), round(varianza, 6), round(desviacion, 6), round(tiempo_total, 2), round(cpu_media, 2), round(memoria_pico, 2)])

        print(f"[+] Guardado: Genético | {iters_totales} evals | Acc: {max_score:.4f} | Tiempo: {tiempo_total:.2f}s")