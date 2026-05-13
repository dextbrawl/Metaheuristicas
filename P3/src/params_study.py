import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prueba as modelo
from population import Population
from individual import Individual
import copy
import os

def create_folders(model_name):
    """Crea la estructura de carpetas para cada modelo"""
    base_path = f'stats_{model_name}'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return base_path

def run_genetic_algorithm_expert(model, params):
    # ngen se toma de params['generations'], que ahora será 200
    pop = Population(size=params['pop_size'], numPoints=20, limits=(-1.0, 1.0), model=model)
    pop.initializeRandom()
    pop.evaluateAll()
    
    stats_history = {
        'best': [], 'avg': [], 'worst': [], 'std': [],
        'diversity': [], 'success_rate': []
    }
    
    global_best_ind = None

    for gen in range(params['generations']):
        current_stats = pop.getStatistics()
        stats_history['best'].append(current_stats['min'])
        stats_history['avg'].append(current_stats['avg'])
        stats_history['worst'].append(current_stats['max'])
        stats_history['std'].append(current_stats['std'])
        
        # Diversidad: Distancia al centroide
        all_coords = np.array([ind.points.flatten() for ind in pop.individuals])
        centroid = np.mean(all_coords, axis=0)
        dist_to_centroid = np.mean([np.linalg.norm(c - centroid) for c in all_coords])
        stats_history['diversity'].append(dist_to_centroid)

        sorted_pop = sorted(pop.individuals, key=lambda x: x.fitness)
        
        if global_best_ind is None or sorted_pop[0].fitness < global_best_ind.fitness:
            global_best_ind = copy.deepcopy(sorted_pop[0])

        new_indivs = []
        for i in range(params['elite_size']):
            source = sorted_pop[i]
            elite = Individual(numPoints=20, limits=(-1.0, 1.0), model=model)
            elite.points, elite.pairs, elite.fitness = source.points.copy(), list(source.pairs), source.fitness
            elite.classes = source.classes.copy() if source.classes is not None else None
            new_indivs.append(elite)
            
        while len(new_indivs) < params['pop_size']:
            p1, p2 = pop.tournamentSelection(3), pop.tournamentSelection(3)
            c1, c2 = pop.crossing(p1, p2)
            new_indivs.extend([c1, c2])
        
        pop.individuals = new_indivs[:params['pop_size']]
        pop.mutation(params['mut_prob'], params['mut_rate'], eliteSize=params['elite_size'])
        pop.evaluateAll()
        
        improvements = sum(1 for ind in pop.individuals if ind.fitness < current_stats['avg'])
        stats_history['success_rate'].append(improvements / params['pop_size'])
        
    return stats_history, global_best_ind

def save_plot(filename, path):
    plt.savefig(os.path.join(path, filename))
    plt.close()

def plot_param_convergence(model, param_name, values, base_params, path):
    plt.figure(figsize=(10, 6))
    for val in values:
        p = base_params.copy()
        p[param_name] = val
        res, _ = run_genetic_algorithm_expert(model, p)
        plt.plot(res['best'], label=f"{param_name}={val}")
    plt.title(f"Convergencia según {param_name} (200 Gens)")
    plt.xlabel("Generación")
    plt.ylabel("Mejor Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(f"conv_{param_name}.png", path)

def plot_summary_stats(results, gens, path):
    plt.figure(figsize=(10, 6))
    plt.plot(gens, results['best'], label='Mejor', color='blue')
    plt.plot(gens, results['avg'], label='Promedio', color='green')
    plt.fill_between(gens, results['best'], results['worst'], alpha=0.1, color='blue')
    plt.title("Evolución General del Fitness (200 Gens)")
    plt.legend()
    save_plot("01_evolucion.png", path)

    plt.figure(figsize=(10, 6))
    plt.plot(gens, results['diversity'], color='purple')
    plt.title("Diversidad Genética (Centroide)")
    save_plot("02_diversidad.png", path)

    plt.figure(figsize=(10, 6))
    plt.bar(gens, results['success_rate'], color='orange', alpha=0.6)
    plt.title("Eficiencia de Operadores (Success Rate)")
    save_plot("03_tasa_mejora.png", path)

def plot_final_demo(model, ind, path):
    res = 150
    x = np.linspace(-1.0, 1.0, res)
    y = np.linspace(-1.0, 1.0, res)
    X, Y = np.meshgrid(x, y)
    grid = np.c_[X.ravel(), Y.ravel()]
    Z = np.array([model.predict(p) for p in grid]).reshape(X.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, alpha=0.3, cmap='RdBu')
    pts = ind.points
    cls = ind.getClasses(model)
    for i, j in ind.pairs:
        plt.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], 'k-', alpha=0.2)
        for idx in [i, j]:
            plt.scatter(pts[idx,0], pts[idx,1], c='red' if cls[idx]==1 else 'blue', 
                        marker='s' if cls[idx]==1 else 'o', edgecolors='black', s=80, zorder=5)
    plt.title(f"Mapeo Final de Frontera (Fitness: {ind.fitness:.4f})")
    save_plot("05_frontera_final.png", path)

if __name__ == "__main__":
    modelos_archivos = ["blackbox_modelA.pkl", "blackbox_modelB.pkl"]
    
    # MODIFICACIÓN: Generations aumentada a 200
    base_params = {
        'pop_size': 50, 
        'mut_prob': 0.3, 
        'mut_rate': 0.2, 
        'elite_size': 2, 
        'generations': 200 
    }

    for arch in modelos_archivos:
        name = arch.split('.')[0]
        print(f"\n>>> INICIANDO ESTUDIO EXTENDIDO (200 GENS) PARA: {name}")
        path = create_folders(name)
        bb_model = modelo.BlackBoxModel(arch)
        
        print(f"[{name}] Estudiando parámetros de mutación...")
        plot_param_convergence(bb_model, 'mut_prob', [0.1, 0.4, 0.7], base_params, path)
        plot_param_convergence(bb_model, 'mut_rate', [0.05, 0.2, 0.5], base_params, path)
        
        print(f"[{name}] Generando estadísticas detalladas...")
        results, top_ind = run_genetic_algorithm_expert(bb_model, base_params)
        plot_summary_stats(results, range(base_params['generations']), path)
        
        print(f"[{name}] Generando mapa de frontera final...")
        plot_final_demo(bb_model, top_ind, path)

    print("\nPROCESO COMPLETADO. Revisa las carpetas 'stats_blackbox_modelA' y 'stats_blackbox_modelB'.")