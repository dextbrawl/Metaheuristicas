import random
import metrics as me
import math

# En algunas funciones de enfriamiento, se pasan valores no usados para estandarizar las funciones
# podiendo así pasarlas como parámetro.

def linealCooling(initialTemperature, finalTemperature, i, max_iter):
    beta = (initialTemperature - finalTemperature) / max_iter
    temperature = initialTemperature - i * beta
    return temperature

def logarithmCooling(initialTemperature, i, finalTemperature, max_iter):
    T = initialTemperature / (1 + math.log(i))
    return T

# M es max_iter, el número máximo de iteraciones que queremos que corra el algoritmo.
def geometricCooling(initialTemperature, i, finalTemperature, max_iter):
    alpha = (finalTemperature / initialTemperature) ** (1 / max_iter)
    T = (alpha ** i) * initialTemperature
    return T

def cauchyCooling(initialTemperature, i, finalTemperature, max_iter):
    T = initialTemperature / (1 + i)
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


def simmulatedAnnealing(series: list, k_segments: int, T0: float, L: int, Tf: float, coolingFunction):
    print("-- SIMMULATED ANNEALING --")
    
    size = len(series)
    initial_bp = me.getBreakingPoints(size, k_segments)
    initial_mse = me.avgMSE(series, initial_bp)
    
    best_bp = initial_bp
    best_mse = initial_mse

    step_size = int(0.01 * size)
    
    T = T0
    i = 0
    errors = []
    errors.append(initial_mse)

    while T >= Tf:
        for count in range(L):
            s_cand = generateNeighbour(initial_bp, step_size)
            new_mse = me.avgMSE(series, s_cand)
            errors.append(new_mse)

            delta = new_mse - initial_mse
            U = random.random()
            exponent = (-delta / T)
            try:
                probability = math.exp(exponent)
            except OverflowError:
                probability = 0 
                
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

        i += 1
        if i >= max_iter:
            break
        
        T = coolingFunction(T0, i, Tf, max_iter)
        print(f"TEMPERATURE: ", T)

    errors_mean = me.calculateErrorMean(errors)
    print(f"Average of errors: ", errors_mean)
    
    if len(errors) > 1:
        error_variance = me.calculateVariance(errors)
        print(f"variance of errors: ", error_variance)
        standard_desviation = me.calculateStandardDesviation(errors)
        print(f"Standard desviation of errors: ", standard_desviation)
    else:
        print("variance of errors: 0.0")
        print("Standard desviation of errors: 0.0")

    return best_bp

if __name__ == '__main__':
    filename, k_segments = me.select_series()

    series = me.readSeries(filename)

    sol = simmulatedAnnealing(series, k_segments, 100, 30, 0.01, geometricCooling)

    sol_bp = sol[0]
    sol_mse = sol[1]

    print("THE SOLUTION IS: \n")
    print(f"BEST AVERAGE MSE: ", sol_mse)
    print(f"BEST BREAKING POINTS: ", sol_bp)
    me.draw(series, sol_bp, filename)
