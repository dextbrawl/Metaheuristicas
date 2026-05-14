import numpy as np
import matplotlib.pyplot as plt
import prueba as modelo
from population import Population
from individual import Individual
from scipy.spatial import ConvexHull

#FUNCIONES DE MATPLOT HECHAS CON IA CASI ENTERAS :)

def findNearest(points, currentPoint):
    """
    Dado un conjunto de puntos y un punto, haya el punto con menor distancia euclidiana al actual (currentPoint)
    """

    minDist = 4000
    
    retVal = currentPoint

    for p in points:
        dist = np.linalg.norm(currentPoint - p)
        if dist < minDist and dist > 0:
            minDist = dist
            retVal = p
    return retVal

def drawIndividual(individual, model, title="Mejor Individuo"):
    """
    Dibuja los puntos de un individuo en R2 coloreados por clase
    Args:
        individual: El individuo a dibujar
        model: Modelo black box
        title: Título del gráfico
    """
    if individual.classes is None:
        individual.getClasses(model)
    
    # Separar puntos por clase
    points_class0 = []
    points_class1 = []
    
    for i, point in enumerate(individual.points):
        if individual.classes[i] == 0:
            points_class0.append(point)
        else:
            points_class1.append(point)
    
    points_class0 = np.array(points_class0)
    points_class1 = np.array(points_class1)
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    
    # Dibujar puntos de clase 0
    if len(points_class0) > 0:
        plt.scatter(points_class0[:, 0], points_class0[:, 1], 
                   c='blue', marker='o', s=80, edgecolors='black', 
                   linewidth=1.5, label='Clase 0', alpha=0.8)
    
    # Dibujar puntos de clase 1
    if len(points_class1) > 0:
        plt.scatter(points_class1[:, 0], points_class1[:, 1], 
                   c='red', marker='s', s=80, edgecolors='black', 
                   linewidth=1.5, label='Clase 1', alpha=0.8)
    
    # Dibujar las líneas de los pares
    if individual.pairs:
        for i, j in individual.pairs:
            x_coords = [individual.points[i][0], individual.points[j][0]]
            y_coords = [individual.points[i][1], individual.points[j][1]]
            #plt.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1, linestyle='--')
    
    plt.xlabel('Eje X', fontsize=12)
    plt.ylabel('Eje Y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Añadir información del fitness
    if individual.fitness is not None:
        plt.text(0.02, 0.98, f'Fitness: {individual.fitness:.4f}', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def drawDecisionBoundary(individual, model, limits=(-1.0, 1.0), title="Mejor Individuo", resolution=100):
    """
    Dibuja la frontera de decisión del modelo junto con los puntos del individuo
    """
    if individual.classes is None:
        individual.getClasses(model)
    
    # Crear malla para visualizar la frontera
    x = np.linspace(limits[0], limits[1], resolution)
    y = np.linspace(limits[0], limits[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluar modelo en la malla
    Z = np.zeros(X.shape)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = model.predict([X[i, j], Y[i, j]])
    
    # Separar puntos del individuo por clase
    points_class0 = []
    points_class1 = []
    
    for i, point in enumerate(individual.points):
        if individual.classes[i] == 0:
            points_class0.append(point)
        else:
            points_class1.append(point)
    
    points_class0 = np.array(points_class0)
    points_class1 = np.array(points_class1)
    
    # Crear figura
    plt.figure(figsize=(12, 10))
    
    # Frontera de decisión
    plt.contourf(X, Y, Z, levels=[-0.5, 0.5, 1.5], 
                 colors=['#3498db', '#e74c3c'], alpha=0.3)
    
    # Dibujar frontera aproximada
    aprox = individual.getAproximationPoints()
    if len(aprox) >= 3:
        centroid = np.mean(aprox, axis=0)
        angles = np.arctan2(aprox[:,1] - centroid[1], aprox[:,0] - centroid[0])
        sorted_idx = np.argsort(angles)
        ordered = aprox[sorted_idx]
        # Close the polygon
        ordered = np.vstack([ordered, ordered[0]])
        plt.plot(ordered[:,0], ordered[:,1], 'g-', linewidth=2, label='Aprox. boundary')
    
    # Dibujar puntos del individuo
    if len(points_class0) > 0:
        plt.scatter(points_class0[:, 0], points_class0[:, 1], 
                   c='blue', marker='o', s=100, edgecolors='white', 
                   linewidth=1.5, label='Puntos Clase 0', alpha=0.9, zorder=3)
    
    if len(points_class1) > 0:
        plt.scatter(points_class1[:, 0], points_class1[:, 1], 
                   c='red', marker='s', s=100, edgecolors='white', 
                   linewidth=1.5, label='Puntos Clase 1', alpha=0.9, zorder=3)
    
    # Dibujar los pares
    if individual.pairs:
        for i, j in individual.pairs:
            x_coords = [individual.points[i][0], individual.points[j][0]]
            y_coords = [individual.points[i][1], individual.points[j][1]]
            plt.plot(x_coords, y_coords, 'gray', alpha=0.6, linewidth=1.5, linestyle='--')
    
    plt.xlabel('Eje X', fontsize=12)
    plt.ylabel('Eje Y', fontsize=12)
    plt.title(f'Frontera de Decision - {title}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if individual.fitness is not None:
        plt.text(0.02, 0.98, f'Fitness: {individual.fitness:.4f}', 
                transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

#Grafica de evolucion de la funcion objetivo segun va evolucionando
def drawFitnessEvolution():
    generations = range(len(bestHistory))

    plt.figure(figsize=(12, 6))

    plt.plot(generations, bestHistory, 'b-', linewidth=2, label='Mejor fitness')
    plt.plot(generations, avgHistory, 'g-', linewidth=2, label='Fitness promedio')
    plt.plot(generations, worstHistory, 'r-', linewidth=2, label='Peor fitness', alpha=0.7)

    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Evolución del Fitness - Algoritmo Genético', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Marcar el mejor valor encontrado
    bestValue = min(bestHistory)
    bestGen = np.argmin(bestHistory)
    plt.plot(bestGen, bestValue, 'ro', markersize=10, label=f'Mejor global: {bestValue:.4f}')
    plt.annotate(f'{bestValue:.4f}', (bestGen, bestValue), 
                xytext=(10, 10), textcoords='offset points')

    plt.tight_layout()
    #Esta linea pa guardarlo
    #plt.savefig('evolucion_fitness.png', dpi=150)
    plt.show()

#Nota: Hay que hacerlo para los dos modelos de la practica
model = modelo.BlackBoxModel("blackbox_modelA.pkl")

#Parametrillos del algoritmo

POPULATION_SIZE = 50
NUM_POINTS = 100
LIMITS = (-4.0, 4.0)
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.85
MUTATION_PROB = 0.3
MUTATION_RATE = 0.2
GENERATIONS = 200

#I. Iniciamos pobacion random
print(f"\nCreando poblacion inicial de {POPULATION_SIZE} individuos...")
currentPopulation = Population(
    size=POPULATION_SIZE,
    numPoints=NUM_POINTS,
    limits=LIMITS,
    model=model
)
currentPopulation.initializeRandom()
currentPopulation.evaluateAll()

# Guardar historiales
bestHistory = []
avgHistory = []
worstHistory = []

# Guardar el mejor individuo global
globalBest = currentPopulation.getBest()
globalBestFitness = globalBest.fitness
globalBestIndividual = globalBest

stats = currentPopulation.getStatistics()
bestHistory.append(stats['min'])
avgHistory.append(stats['avg'])
worstHistory.append(stats['max'])

print("\nEvolucionando generaciones...")

#II. Nuevas generaciones
for generation in range(GENERATIONS):
    newPopulation = Population(
        size=POPULATION_SIZE,
        numPoints=NUM_POINTS,
        limits=LIMITS,
        model=model
    )
    newPopulation.individuals = []
    
    # Elite
    sortedIndividuals = sorted(currentPopulation.individuals, key=lambda ind: ind.fitness)
    for i in range(min(ELITE_SIZE, len(sortedIndividuals))):
        source = sortedIndividuals[i]
        elite = Individual(numPoints=NUM_POINTS, limits=LIMITS, model=model)
        
        elite.points = source.points.copy()
        elite.pairs = list(source.pairs)  # ¡ESTO ES CLAVE! Copia las parejas
        elite.fitness = source.fitness
        elite.classes = source.classes.copy() if source.classes is not None else None
        elite.components = source.components.copy()
        
        newPopulation.individuals.append(elite)
    
    # Cruce
    while len(newPopulation.individuals) < POPULATION_SIZE:
        parent1 = currentPopulation.tournamentSelection(TOURNAMENT_SIZE)
        parent2 = currentPopulation.tournamentSelection(TOURNAMENT_SIZE)
        
        if np.random.random() < CROSSOVER_PROB:
            child1, child2 = currentPopulation.crossing(parent1, parent2)
        else:
            child1 = Individual(numPoints=NUM_POINTS, limits=LIMITS, model=model)
            child1.points = parent1.points.copy()
            
            child2 = Individual(numPoints=NUM_POINTS, limits=LIMITS, model=model)
            child2.points = parent2.points.copy()
        
        newPopulation.individuals.append(child1)
        if len(newPopulation.individuals) < POPULATION_SIZE:
            newPopulation.individuals.append(child2)
    

    #MUTAR INDIVIDUOS

    newPopulation.mutation(MUTATION_PROB, MUTATION_RATE, eliteSize=ELITE_SIZE)

    

    #Nueva gen...
    newPopulation.evaluateAll()
    currentPopulation = newPopulation

    
    
    stats = currentPopulation.getStatistics()
    bestHistory.append(stats['min'])
    avgHistory.append(stats['avg'])
    worstHistory.append(stats['max'])
    
    # Actualizar
    currentBest = currentPopulation.getBest()
    if currentBest.fitness < globalBestFitness:
        globalBestFitness = currentBest.fitness
        globalBestIndividual = currentBest
    
    # Mostrar progreso
    if generation % 5 == 0:
        print(f"   Gen {generation:3d}: Mejor={stats['min']:.4f} | "
              f"Promedio={stats['avg']:.4f} | Peor={stats['max']:.4f}")

print("\nFIN")

print(f"Mejor panchito: {globalBestFitness:.4f}")
print(globalBestIndividual.points)

if hasattr(globalBestIndividual, 'components'):
    print("\nComponentes del mejor individuo:")
    print(f"\nDistancia promedio entre pares: {globalBestIndividual.components.get('avgDistance', 0):.4f}")
    print(f"\nDispersion: {globalBestIndividual.components.get('dispersion', 0):.4f}")
    print(f"\nPenalizacion variedad: {globalBestIndividual.components.get('varietyPenalty', 0):.4f}")
    print(f"\nPenalizacion misma clase: {globalBestIndividual.components.get('sameClassPenalty', 0):.4f}")



drawFitnessEvolution()
globalBestIndividual.minpairdistance()
drawIndividual(globalBestIndividual, model, title="Top Individuo")
drawDecisionBoundary(globalBestIndividual, model, limits=LIMITS, title="Top Individuo")
