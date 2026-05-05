import math
import operator
import random as rnd
from os import name

import numpy as np
from deap import algorithms, base, creator, gp, tools


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
pset.addPrimitive(secureDiv, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

# Generamos constantes para los nodos hojas de los árboles
pset.addEphemeralConstant("rand101", lambda: rnd.uniform(-10, 10))

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
toolbox.register("select", tools.selTournament, tournsize=3)


def evalIndividual(individual, points):
    func = toolbox.compile(expr=individual)
    acc_mse = 0

    for x, y in points:
        try:
            result = (func(x, y)) ** 2
            acc_mse += result
        except OverflowError:
            # Si una ecuación se vuelve infinita, da la máxima penalización
            return float("inf")

    avg_mse = acc_mse / len(points)

    tam_penalty = 0.01 * len(individual)

    final_fitness_val = avg_mse + tam_penalty

    # Se devuelve de esta forma ya que DEAP espera normalmente que el fitness devuelto sea una tupla
    return (final_fitness_val,)


# variable donde estarán los puntos obtenidos para la regresión simbólica
points = []
# Registramos la función de evaluación
toolbox.register("evaluate", evalIndividual, puntos_frontera=points)

# Algoritmo evolutivo

population = toolbox.population(n=300)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, ngen=50, verbose=True)
bestOne = tools.selBest(population, 1)[0]

print("La mejor ecuación encontrada es: ", bestOne)
