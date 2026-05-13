import functools
import math
import operator
import random as rnd
from bisect import bisect
import os

import joblib
import numpy as np
from deap import algorithms, base, creator, gp, tools
from matplotlib import pyplot as plt

class BlackBoxModel:
    def __init__(self, path="blackbox_model.pkl"):
        if not os.path.exists(path):
            path = "../src/blackbox_modelB.pkl" 
        self.model = joblib.load(path)

    def predict(self, x):
        x = np.array(x).reshape(1, -1)
        return self.model.predict(x)[0]

model = BlackBoxModel("../src/blackbox_modelB.pkl") 

print(f"========== MODELO B IMPORTADO EXITOSAMENTE ==========")

def getInterPoint():
    th_intrap_distance = 0.2
    valid = False
    final_point = None
    while not valid:
        x_1, y_1 = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
        x_2, y_2 = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
        point1, point2 = (x_1, y_1), (x_2, y_2)
        
        class_p1 = model.predict(point1)
        class_p2 = model.predict(point2)
        intrap_distance = math.dist(point1, point2)
        
        if (th_intrap_distance >= intrap_distance) and (class_p1 != class_p2):
            x_bisec = (x_1 + x_2) / 2.00
            y_bisec = (y_1 + y_2) / 2.00
            final_point = (x_bisec, y_bisec)
            valid = True
    return final_point

def compareInterDistance(point, cloud):
    th_interp_distance = 0.15
    for cloud_p in cloud:
        if math.dist(point, cloud_p) < th_interp_distance:
            return False
    return True

def getCloud():
    n_points = 15
    cloud = []
    i = 0
    while i < n_points:
        point = getInterPoint()
        if compareInterDistance(point, cloud):
            cloud.append(point)
            print(f"PUNTO ENCONTRADO {i + 1}: {point}")
            i += 1
    return cloud

pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="x", ARG1="y")
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addEphemeralConstant("rand101", functools.partial(rnd.uniform, -1, 1))

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
    rx, ry = max_x - min_x, max_y - min_y
    mx, my = rx * 0.20, ry * 0.20
    return [
        (min_x - mx, min_y - my), (max_x + mx, max_y + my),
        (min_x - mx, max_y + my), (max_x + mx, min_y - my),
        (min_x + rx/2, min_y + ry/2)
    ]

def evalIndividual(individual, points, contrast_points):
    if "x" not in str(individual) or "y" not in str(individual):
        return (float("inf"),)
    
    func = toolbox.compile(expr=individual)
    acc_mse, acc_contrast = 0, 0
    min_res, max_res = float("inf"), float("-inf")

    try:
        for x, y in points:
            res = func(x, y)
            min_res, max_res = min(min_res, res), max(max_res, res)
            acc_mse += res**2
        
        for cx, cy in contrast_points:
            res = func(cx, cy)
            min_res, max_res = min(min_res, res), max(max_res, res)
            if abs(res) < 0.1: return (float("inf"),)
            acc_contrast += 1.0 / (res**2)
            
    except (OverflowError, ZeroDivisionError):
        return (float("inf"),)

    if (max_res - min_res) < 1e-5: return (float("inf"),)

    avg_mse = (acc_mse / len(points)) * 1000.0
    avg_contrast = acc_contrast / len(contrast_points)
    size_penalty = 0.07 * len(individual)

    return (avg_mse + avg_contrast + size_penalty,)

if __name__ == "__main__":
    points = getCloud()
    contrast_points = getDynamicContrastPoints(points)
    toolbox.register("evaluate", evalIndividual, points=points, contrast_points=contrast_points)

    def valid_fits(fits):
        return [f for f in fits if f < float('inf') and not np.isnan(f)]

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", lambda fits: np.min(valid_fits(fits)) if valid_fits(fits) else float('inf'))
    stats.register("avg", lambda fits: np.mean(valid_fits(fits)) if valid_fits(fits) else float('inf'))
    stats.register("max", lambda fits: np.max(valid_fits(fits)) if valid_fits(fits) else float('inf'))
    stats.register("std", lambda fits: np.std(valid_fits(fits)) if valid_fits(fits) else 0.0)

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=1000, 
                                   stats=stats, halloffame=hof, verbose=True)

    bestOne = hof[0]
    print("\nMejor ecuación histórica: ", bestOne)

    func = toolbox.compile(expr=bestOne)
    x_v = np.linspace(-1.5, 1.5, 400)
    y_v = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x_v, y_v)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try: Z[i, j] = func(X[i, j], Y[i, j])
            except: Z[i, j] = np.nan

    plt.figure(figsize=(8, 8))
    try: plt.contour(X, Y, Z, levels=[0], colors="red", linewidths=2)
    except: print("No hay nivel 0 visible.")
    plt.scatter([p[0] for p in points], [p[1] for p in points], c="blue", s=20)
    plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5); plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    print(f"FITNESS_FINAL:{bestOne.fitness.values[0]}")
    print(f"STD_FINAL:{log.select('std')[-1]}")