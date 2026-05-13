import functools
import math
import operator
import random as rnd
from bisect import bisect
from os import name
import os

import joblib
import numpy as np
from deap import algorithms, base, creator, gp, tools
from matplotlib import pyplot as plt


class BlackBoxModel:
    def __init__(self, path="blackbox_modelB.pkl"):
        # Aseguramos manejar bien el path relativo o directo
        if not os.path.exists(path):
            path = "../" + path if os.path.exists("../" + path) else path
        self.model = joblib.load(path)

    def predict(self, x):
        x = np.array(x).reshape(1, -1)
        return self.model.predict(x)[0]


model = BlackBoxModel("blackbox_modelA.pkl")
print(f"========== MODELO IMPORTADO EXITOSAMENTE ==========")


def getInterPoint():
    th_intrap_distance = 0.4
    valid = False
    final_point = None
    while not valid:
        x_1, y_1 = rnd.uniform(-4, 4), rnd.uniform(-4, 4)
        x_2, y_2 = rnd.uniform(-4, 4), rnd.uniform(-4, 4)
        point1 = (x_1, y_1)
        point2 = (x_2, y_2)
        class_p1 = model.predict(point1)
        class_p2 = model.predict(point2)
        intrap_distance = math.dist(point1, point2)
        if (th_intrap_distance >= intrap_distance) and (class_p1 != class_p2):
            x_bisec, y_bisec = (x_1 + x_2) / 2.00, (y_1 + y_2) / 2.00
            final_point = (x_bisec, y_bisec)
            valid = True
    return final_point


def compareInterDistance(point, cloud):
    th_interp_distance = 0.3
    for cloud_p in cloud:
        point_dist = math.dist(point, cloud_p)
        if point_dist < th_interp_distance:
            return False
    return True


def getCloud():
    n_points = 35
    cloud = []
    i = 0
    point = getInterPoint()
    cloud.append(point)
    print(f"PRIMER PUNTO: {point}")
    while i < n_points:
        point = getInterPoint()
        if compareInterDistance(point, cloud):
            cloud.append(point)
            print(f"PUNTO ENCONTRADO {i + 1}: {point}")
            i += 1
    print(f"ESTA ES LA NUBE GENERADA: {cloud}")
    return cloud


def secureDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0


pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="x", ARG1="y")
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addEphemeralConstant("rand101", functools.partial(rnd.uniform, -10, 10))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)


def getDynamicContrastPoints(boundary_points):
    x_coords = [p[0] for p in boundary_points]
    y_coords = [p[1] for p in boundary_points]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    range_x, range_y = max_x - min_x, max_y - min_y
    margin_x, margin_y = range_x * 0.20, range_y * 0.20

    return [
        (min_x - margin_x, min_y - margin_y),
        (max_x + margin_x, max_y + margin_y),
        (min_x - margin_x, max_y + margin_y),
        (max_x + margin_x, min_y - margin_y),
        (min_x + range_x / 2.0, min_y + range_y / 2.0)
    ]


def evalIndividual(individual, points, contrast_points):
    tree_str = str(individual)
    if "x" not in tree_str or "y" not in tree_str:
        return (float("inf"),)

    func = toolbox.compile(expr=individual)
    acc_mse = 0
    acc_contrast = 0
    min_res = float("inf")
    max_res = float("-inf")

    for x, y in points:
        try:
            res = func(x, y)
            min_res = min(min_res, res)
            max_res = max(max_res, res)
            acc_mse += res**2
        except OverflowError:
            return (float("inf"),)

    for cx, cy in contrast_points:
        try:
            res = func(cx, cy)
            min_res = min(min_res, res)
            max_res = max(max_res, res)
            if abs(res) < 0.1:
                return (float("inf"),)
            acc_contrast += 1.0 / (res**2)
        except OverflowError:
            return (float("inf"),)

    value_range = max_res - min_res
    if value_range < 1e-5:
        return (float("inf"),)

    avg_mse = (acc_mse / len(points)) * 1000.0
    avg_contrast = acc_contrast / len(contrast_points)
    size_penalty = 0.07 * len(individual)

    final_fitness_val = avg_mse + avg_contrast + size_penalty
    return (final_fitness_val,)


# ----------- FUNCIONES DE GRÁFICAS -----------
def draw_fitness_evolution(logbook):
    gen = logbook.select("gen")
    fit_min = logbook.select("min")
    fit_avg = logbook.select("avg")
    fit_max = logbook.select("max")

    plt.figure(figsize=(12, 6))
    plt.plot(gen, fit_min, 'b-', linewidth=2, label='Mejor Fitness')
    plt.plot(gen, fit_avg, 'g-', linewidth=2, label='Fitness Promedio')
    plt.plot(gen, fit_max, 'r-', linewidth=1, label='Peor Fitness', alpha=0.5)
    
    # Rellenar el área entre el mejor y el peor para ver la varianza
    plt.fill_between(gen, fit_min, fit_max, alpha=0.1, color='blue')

    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness (Escala Logarítmica)', fontsize=12)
    plt.title('Evolución del Fitness - Programación Genética', fontsize=14)
    plt.yscale('log') # Útil en GP ya que los fitness al inicio suelen explotar
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gp_evolucion_fitness.png', dpi=150)
    print("\n[+] Gráfica de evolución guardada como 'gp_evolucion_fitness.png'")
    plt.show()

# ----------- EJECUCIÓN PRINCIPAL -----------
if __name__ == "__main__":
    points = getCloud()
    contrast_points = getDynamicContrastPoints(points)
    
    toolbox.register("evaluate", evalIndividual, points=points, contrast_points=contrast_points)

    # Configuramos las estadísticas (excluimos los infinitos para no distorsionar las gráficas)
    def valid_fits(fits):
        return [f for f in fits if f < float('inf')]

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", lambda fits: np.min(valid_fits(fits)) if valid_fits(fits) else float('inf'))
    stats.register("avg", lambda fits: np.mean(valid_fits(fits)) if valid_fits(fits) else float('inf'))
    stats.register("max", lambda fits: np.max(valid_fits(fits)) if valid_fits(fits) else float('inf'))
    stats.register("std", lambda fits: np.std(valid_fits(fits)) if valid_fits(fits) else 0.0)

    population = toolbox.population(n=500)
    
    print("\n========== INICIANDO ALGORITMO EVOLUTIVO (GP) ==========")
    # Nota: He reducido ngen a 100 para que termine más rápido durante las pruebas. 
    # Vuelve a ponerlo a 1000 cuando quieras tu modelo definitivo.
    NGEN = 100
    
    # eaSimple ahora nos devuelve también el Logbook con las estadísticas recopiladas
    population, logbook = algorithms.eaSimple(
        population, toolbox, 
        cxpb=0.7, mutpb=0.3, 
        ngen=NGEN, 
        stats=stats, 
        verbose=True
    )
    
    bestOne = tools.selBest(population, 1)[0]
    print("\nLa mejor ecuación encontrada es: ", bestOne)

    # Generamos la gráfica de evolución
    draw_fitness_evolution(logbook)

    # Generamos la gráfica de la Frontera Evolutiva (Tu código original adaptado)
    func = toolbox.compile(expr=bestOne)
    x_val = np.linspace(-5, 5, 400)
    y_val = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x_val, y_val)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = func(X[i, j], Y[i, j])
            except (ZeroDivisionError, OverflowError):
                Z[i, j] = np.nan

    plt.figure(figsize=(8, 8))
    plt.title("Renderizado de la Frontera Evolutiva (Mejor Ecuación)", fontsize=14)

    try:
        contour = plt.contour(X, Y, Z, levels=[0], colors="red", linewidths=2)
    except ValueError:
        print("\n[!] Ojo: La ecuación no tiene un valor = 0 en el rango visible de -5 a 5.")

    puntos_x = [p[0] for p in points]
    puntos_y = [p[1] for p in points]
    plt.scatter(puntos_x, puntos_y, c="blue", s=20, label="Puntos Bisección")

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig("gp_frontera_final.png", dpi=150)
    print("[+] Gráfica de la frontera guardada como 'gp_frontera_final.png'")
    plt.show()