import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prueba as modelo
from population import Population
from individual import Individual
import copy
import time

def run_genetic_algorithm(model, params):
    pop = Population(
        size=params['pop_size'],
        numPoints=20,
        limits=(-1.0, 1.0),
        model=model
    )
    pop.initializeRandom()
    pop.evaluateAll()
    
    best_per_gen = []
    diversity_per_gen = []
    
    for gen in range(params['generations']):
        sorted_pop = sorted(pop.individuals, key=lambda x: x.fitness)
        best_per_gen.append(sorted_pop[0].fitness)
        
        all_coords = np.array([ind.points.flatten() for ind in pop.individuals])
        avg_dist = np.mean(np.std(all_coords, axis=0))
        diversity_per_gen.append(avg_dist)
        
        new_indivs = []
        for i in range(params['elite_size']):
            source = sorted_pop[i]
            elite = Individual(numPoints=20, limits=(-1.0, 1.0), model=model)
            elite.points = source.points.copy()
            elite.pairs = list(source.pairs)
            elite.fitness = source.fitness
            elite.classes = source.classes.copy() if source.classes is not None else None
            elite.components = source.components.copy()
            new_indivs.append(elite)
            
        while len(new_indivs) < params['pop_size']:
            p1 = pop.tournamentSelection(3)
            p2 = pop.tournamentSelection(3)
            if np.random.random() < 0.85:
                c1, c2 = pop.crossing(p1, p2)
            else:
                c1 = Individual(numPoints=20, limits=(-1.0, 1.0), model=model)
                c1.points = p1.points.copy()
                c2 = Individual(numPoints=20, limits=(-1.0, 1.0), model=model)
                c2.points = p2.points.copy()
            new_indivs.append(c1)
            if len(new_indivs) < params['pop_size']:
                new_indivs.append(c2)
        
        pop.individuals = new_indivs
        pop.mutation(params['mut_prob'], params['mut_rate'], eliteSize=params['elite_size'])
        pop.evaluateAll()
        
    return {
        'best_history': best_per_gen,
        'diversity_history': diversity_per_gen,
        'final_fitness': pop.getStatistics()['min']
    }

def plot_parameter_study(param_name, values, base_params, model, iterations=3):
    plt.figure(figsize=(10, 6))
    for val in values:
        current_params = base_params.copy()
        current_params[param_name] = val
        all_runs = [run_genetic_algorithm(model, current_params)['best_history'] for _ in range(iterations)]
        avg_history = np.mean(all_runs, axis=0)
        plt.plot(avg_history, label=f"{param_name}={val}")
    
    plt.title(f"Evolución del Fitness según {param_name}")
    plt.xlabel("Generación")
    plt.ylabel("Mejor Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"study_{param_name}.png")
    plt.show()

def plot_diversity_study(model, base_params):
    result = run_genetic_algorithm(model, base_params)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(result['best_history'], 'b-', label="Mejor Fitness")
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness", color='b')
    
    ax2 = ax1.twinx()
    ax2.plot(result['diversity_history'], 'r--', label="Diversidad (Std Dev)")
    ax2.set_ylabel("Diversidad Genética", color='r')
    
    plt.title("Relación Fitness vs Diversidad")
    plt.grid(True)
    plt.savefig("diversity_analysis.png")
    plt.show()

def plot_robustness_boxplot(model, base_params, iterations=15):
    results = [run_genetic_algorithm(model, base_params)['final_fitness'] for _ in range(iterations)]
    plt.figure(figsize=(8, 6))
    plt.boxplot(results)
    plt.title("Análisis de Robustez (Fitness Final)")
    plt.ylabel("Fitness")
    plt.xticks([1], ['Configuración Base'])
    plt.savefig("robustness_boxplot.png")
    plt.show()

def plot_heatmap_params(model, mut_probs, pop_sizes, base_params):
    grid = np.zeros((len(pop_sizes), len(mut_probs)))
    for i, ps in enumerate(pop_sizes):
        for j, mp in enumerate(mut_probs):
            params = base_params.copy()
            params['pop_size'] = ps
            params['mut_prob'] = mp
            grid[i, j] = run_genetic_algorithm(model, params)['final_fitness']
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid, annot=True, xticklabels=mut_probs, yticklabels=pop_sizes, cmap="YlGnBu")
    plt.title("Heatmap: Pop Size vs Mutation Prob")
    plt.xlabel("Mutation Prob")
    plt.ylabel("Population Size")
    plt.savefig("heatmap_params.png")
    plt.show()

if __name__ == "__main__":
    bb_model = modelo.BlackBoxModel("blackbox_modelB.pkl")
    
    base_params = {
        'pop_size': 50,
        'mut_prob': 0.3,
        'mut_rate': 0.2,
        'elite_size': 2,
        'generations': 50
    }

    plot_parameter_study('mut_prob', [0.1, 0.4, 0.8], base_params, bb_model)
    plot_diversity_study(bb_model, base_params)
    plot_robustness_boxplot(bb_model, base_params)
    plot_heatmap_params(bb_model, [0.1, 0.3, 0.6], [20, 50, 100], base_params)