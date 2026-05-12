import functools
import math
import operator
import random as rnd
from bisect import bisect
from os import name

import joblib
import numpy as np
from deap import algorithms, base, creator, gp, tools
from matplotlib import pyplot as plt


class BlackBoxModel:
    # Considerar blackbox_modelA y blackbox_modelB
    def __init__(self, path="blackbox_model.pkl"):
        self.model = joblib.load(path)

    def predict(self, x):
        x = np.array(x).reshape(1, -1)
        return self.model.predict(x)[0]


model = BlackBoxModel("../blackbox_modelA.pkl")

print(f"========== MODELO IMPORTADO EXITOSAMENTE ==========")


# Función para obtener un par de puntos distanciado
def getInterPoint():
    th_intrap_distance = 0.4
    valid = False
    final_point = None
    while valid == False:
        x_1 = rnd.uniform(-4, 4)
        y_1 = rnd.uniform(-4, 4)
        x_2 = rnd.uniform(-4, 4)
        y_2 = rnd.uniform(-4, 4)
        point1 = (x_1, y_1)
        point2 = (x_2, y_2)
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
    th_interp_distance = 0.3
    for cloud_p in cloud:
        point_dist = math.dist(point, cloud_p)
        if point_dist < th_interp_distance:
            return False
    return True


# Función para obtener la nube de pares de puntos con una distancia umbral entre sí.
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


# Definimos una división segura para evitar la división por 0
def secureDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0


# Definimos el conjunto de primitivas
pset = gp.PrimitiveSet("MAIN", 2)  # 2 argumentos, 'x' e 'y'
pset.renameArguments(ARG0="x", ARG1="y")
pset.addPrimitive(operator.add, 2)
# pset.addPrimitive(secureDiv, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

# Generamos constantes para los nodos hojas de los árboles
pset.addEphemeralConstant("rand101", functools.partial(rnd.uniform, -10, 10))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# genera árboles con profundidad entre 1 y 5
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Convertimos de un arbol genético a función ejecutable
toolbox.register("compile", gp.compile, pset=pset)

# Definimos los operadores genéticos para los árboles
toolbox.register("mate", gp.cxOnePoint)  # Cruza intercambiando ramas
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# Mutación uniforme, muta creando una rama nueva
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# toolbox.register("mutate", gp.mutShrink)
toolbox.register("select", tools.selTournament, tournsize=3)


def getDynamicContrastPoints(boundary_points):
    x_coords = [p[0] for p in boundary_points]
    y_coords = [p[1] for p in boundary_points]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    range_x = max_x - min_x
    range_y = max_y - min_y

    margin_x = range_x * 0.20
    margin_y = range_y * 0.20

    bottom_left = (min_x - margin_x, min_y - margin_y)
    top_right = (max_x + margin_x, max_y + margin_y)
    top_left = (min_x - margin_x, max_y + margin_y)
    bottom_right = (max_x + margin_x, min_y - margin_y)
    center = (min_x + range_x / 2.0, min_y + range_y / 2.0)

    return [bottom_left, top_right, top_left, bottom_right, center]


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
    size_penalty = 0.065 * len(individual)

    final_fitness_val = avg_mse + avg_contrast + size_penalty

    return (final_fitness_val,)


# variable donde estarán los puntos obtenidos para la regresión simbólica
points = getCloud()

contrast_points = getDynamicContrastPoints(points)
print(f"Estos son los puntos generados: {points}")
print(f"Estos son los puntos de contraste: {contrast_points}")
# Registramos la función de evaluación
toolbox.register(
    "evaluate", evalIndividual, points=points, contrast_points=contrast_points
)

# Algoritmo evolutivo

population = toolbox.population(n=500)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, ngen=1000, verbose=True)
bestOne = tools.selBest(population, 1)[0]

print("La mejor ecuación encontrada es: ", bestOne)

# Visualizamos la mejor solución
func = toolbox.compile(expr=bestOne)
x_val = np.linspace(-5, 5, 400)
y_val = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_val, y_val)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        try:
            # Capturamos el valor Z
            Z[i, j] = func(X[i, j], Y[i, j])
        except (ZeroDivisionError, OverflowError):
            # Si en este píxel la ecuación explota, lo marcamos como no válido
            Z[i, j] = np.nan

plt.figure(figsize=(8, 8))
plt.title("Renderizado de la Frontera Evolutiva", fontsize=14)

try:
    contour = plt.contour(X, Y, Z, levels=[0], colors="red", linewidths=2)
except ValueError:
    print("[!] Ojo: La ecuación no tiene un valor = 0 en el rango visible de -5 a 5.")

puntos_x = [p[0] for p in points]
puntos_y = [p[1] for p in points]
plt.scatter(puntos_x, puntos_y, c="blue", s=20, label="Puntos Bisección")

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
