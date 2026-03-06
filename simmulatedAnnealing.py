import random
import metrics as me
import math

def linealCooling(initialTemperature, finalTemperature, i, nIterations):
    beta = (initialTemperature - finalTemperature) / nIterations
    temperature = initialTemperature - i * beta
    return temperature

def logarithmCooling(initialTemperature, i):
    T = initialTemperature / (1 + math.log(i))
    return T

def generateNeighbour(breaking_points, step_size: int):
    neighbour = list(breaking_points)
    tam_s = len(neighbour)
    point_index = random.randint(1, tam_s - 2)
    direction = random.choice([1, -1])
    step =  step_size * direction
    if neighbour[point_index - 1] < (neighbour[point_index] + step) < neighbour[point_index + 1]:
        neighbour[point_index] += step
    else:
        if direction == 1:
            neighbour[point_index] = neighbour[point_index + 1] - 1
        else:
            neighbour[point_index] = neighbour[point_index - 1] + 1

    
    return neighbour


def simmulatedAnnealing(series: list, k_segments: int, T0: float, L: int, Tf: float):
    print("-- SIMMULATED ANNEALING --")
    
    size = len(series)
    initial_bp = me.getBreakingPoints(size, k_segments)
    initial_mse = me.avgMSE(series, initial_bp)
    print("THE CURRENT SOLUTION IS: \n")
    print(f"AVERAGE MSE: ", initial_mse)
    print(f"BREAKING POINTS: ", initial_bp)
    print(f"INITIAL TEMPERATURE: ", T0)

    best_bp = initial_bp
    best_mse = initial_mse

    step_size = int(0.01 * size)

    max_iter = int(input("Introduzca el maximo número de iteraciones que desea realizar: "))

    T = T0

    i = 0
    while T >= Tf:
        for count in range(L):
            s_cand = generateNeighbour(initial_bp, step_size)
            new_mse = me.avgMSE(series, s_cand)
            delta = new_mse - initial_mse
            U = random.random() # número aleatorio entre 0 y 1 ---> U(0, 1)
            exponent = (-delta / T)
            probability = math.exp(exponent)
            accept = False

            if delta < 0:
                accept = True
            elif U < probability:
                accept = True

            if accept:
                initial_bp = s_cand
                initial_mse = new_mse
                if initial_mse < best_mse:
                    best_mse = initial_mse
                    best_bp = list(initial_bp)
            print("THE CURRENT SOLUTION IS: \n")
            print(f"AVERAGE MSE: ", initial_mse)
            print(f"BREAKING POINTS: ", initial_bp)

        i += 1

        # T = linealCooling(T0, Tf, i, max_iter)
        
        if i >= max_iter:
            break
        
        T = logarithmCooling(T0, i)
        print(f"TEMPERATURE: ", T)

    return best_bp, best_mse

if __name__ == '__main__':
    filename, k_segments = me.select_series()

    series = me.readSeries(filename)

    sol = simmulatedAnnealing(series, k_segments, 100, 30, 10.5)

    sol_bp = sol[0]
    sol_mse = sol[1]

    print("THE SOLUTION IS: \n")
    print(f"AVERAGE MSE: ", sol_mse)
    print(f"BEST BREAKING POINTS: ", sol_bp)
    me.draw(series, sol_bp, filename)
