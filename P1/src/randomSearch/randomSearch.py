import os
import sys
aux_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'aux'))
sys.path.append(aux_path)
import metrics as me
import concurrent.futures
import os

"""Algoritmo de búsqueda aleatoria en serie"""
def serialRandomSearch(series: list, k_segments, max_iterations):
    print(" -- RANDOM SEARCH --")
    size = len(series)
    
    breaking_points = me.getBreakingPoints(size,k_segments)
    avg_mse = me.avgMSE(series,breaking_points)
    
    print("SEGMENTS")
    print("Average MSE: ", avg_mse) 
    c = 0
    errors = []
    errors.append(avg_mse)
    
    # Obtiene soluciones aleatorias dado un máximo de iteraciones
    for _ in range(max_iterations):
        new_breaking_points = me.getBreakingPoints(size, k_segments)
        new_avg_mse = me.avgMSE(series,new_breaking_points)
        errors.append(new_avg_mse) #Aqui metemos el NUEVO error 

        # Guarda el mejor resultado de cada iteracion
        if new_avg_mse < avg_mse :
            c = 0
            breaking_points = new_breaking_points
            avg_mse = new_avg_mse
            me.clear_screen()
            print(" -- RANDOM SEARCH --")
            print("Average MSE: ", avg_mse)
        else:
            c = c + 1 #Aqui sumo al contador sin mejora

        if c >= (max_iterations/2):
            me.clear_screen()
            print("STOP:Too may interactions without improving.")
            print(" -- RANDOM SEARCH --")
            print("Average MSE: ", avg_mse)
            break



    #Hago return de los puntos para usarlos en draw
    errors_mean = me.calculateErrorMean(errors)
    print(f"Average of errors: ", errors_mean)
    error_variance = me.calculateVariance(errors)
    print(f"variance of errors: ", error_variance)
    standard_desviation = me.calculateStandardDesviation(errors)
    print(f"Standard desviation of errors: ", standard_desviation)
    return breaking_points

"""Obtiene los datos de la solución obtenida en paralelo en concreto"""
def evalParalelSolutions(args):
    size, k_segments, series = args
    breaking_points = me.getBreakingPoints(size, k_segments)
    mse = me.avgMSE(series, breaking_points)
    return breaking_points, mse

"""Algoritmo de búsqueda aleatoria en paralelo"""
def paralelRandomSearch(series: list, k_segments, max_iterations, batch=5):
    print(" -- PARALEL RANDOM SEARCH --")
    size = len(series)
    
    best_breaking_points = me.getBreakingPoints(size,k_segments)
    best_avg_mse = me.avgMSE(series, best_breaking_points)
    
    cores = os.cpu_count()
    if cores is None:
        n_cores = 1
    else:
        n_cores = cores - 1

    # Usamos la mitad de núcleos del procesador
    n_cores_use = max(1, int(n_cores/2))

    print("Segmentos iniciales: ")
    for i in range(k_segments-1):
        print("     [",best_breaking_points[i],",",best_breaking_points[i+1],"]")
    print("Average MSE: ", best_avg_mse) 
    c = 0
    errors = []
    errors.append(best_avg_mse)
    for _ in range(max_iterations):
        args = [(size, k_segments, series) for _ in range(batch)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores_use) as executor:
            results = list(executor.map(evalParalelSolutions, args))

        for bp, mse in results:
            errors.append(mse)

        bestOne_in_batch = min(results, key=lambda x: x[1])
        bests_bp_in_batch, best_mse_batch = bestOne_in_batch

        if best_mse_batch < best_avg_mse :
            c = 0
            best_breaking_points = bests_bp_in_batch
            best_avg_mse = best_mse_batch
            me.clear_screen()
            print(" -- PARALEL RANDOM SEARCH --")
           
            print("SEGMENTS")
            for i in range(k_segments-1):
                print("     [",best_breaking_points[i],",",best_breaking_points[i+1],"]")
            print("Average MSE: ", best_avg_mse)

        else:        
            c = c + 1


        # Condición de parada 
        if c >= (max_iterations/2):
            me.clear_screen()
            print("STOP:Too may interactions without improving.")
            print(" -- PARALEL RANDOM SEARCH --")
           
            print("SEGMENTS")
            for i in range(k_segments-1):
                print("     [",best_breaking_points[i],",",best_breaking_points[i+1],"]")
            print("Average MSE: ", best_avg_mse)

            break 


    # Obtención de datos
    errors_mean = me.calculateErrorMean(errors)
    print(f"Average of errors: ", errors_mean)
    error_variance = me.calculateVariance(errors)
    print(f"variance of errors: ", error_variance)
    standard_desviation = me.calculateStandardDesviation(errors)
    print(f"Standard desviation of errors: ", standard_desviation)
    return best_breaking_points 
#Definimos para nuestra practica, "Fichero": K
if __name__ == '__main__':

    filename, k_segments = me.select_series()

    series = me.readSeries(filename)

    best_breaking_points = paralelRandomSearch(me.readSeries(filename),k_segments, 200)
    me.draw(series,best_breaking_points, filename)
