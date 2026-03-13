import sys
import os

# Para poder importar el módulo desde carpetas externas
aux_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'aux'))
sys.path.append(aux_path)

import metrics as me

"""Obtiene todos los vecinos dado un valor de step para luego analizarlos todos"""
def neighbourhood(breaking_points, step):
    neighbourhood = []
    
    for i in range(1, len(breaking_points) - 1): #El primero y el ultimo no se pueden alterar
        for s in range(1, step + 1):
            if breaking_points[i] + s < breaking_points[i+1]:
                new_points = breaking_points.copy()
                new_points[i] += s
                neighbourhood.append(new_points)
        
        for s in range(1, step + 1):
            if breaking_points[i] - s > breaking_points[i-1]:
                new_points = breaking_points.copy()
                new_points[i] -= s
                neighbourhood.append(new_points)
    
    return neighbourhood

"""Funcionamiento de hill climbing estudiado en clase"""
def hillClimbingSearch(series, k_segments, prev_breaking_points):
    errors = []
    improved = True
    n = int(len(series))
    step = int(0.01 * n) 
    best_breaking_points = prev_breaking_points.copy()
    best_MSE = me.avgMSE(series, prev_breaking_points)
    
    errors.append(best_MSE)
    
    # Hill climbing busca para todos los vecinos hasta que no encuentre ninguno mejor y se atasque en un óptimo
    while improved:
        nbh = neighbourhood(best_breaking_points, step)
        improved = False
        for a in nbh:
            curr_MSE = me.avgMSE(series, a)
            
            errors.append(curr_MSE) 
            
            if curr_MSE < best_MSE:
                best_breaking_points = a.copy()
                best_MSE = curr_MSE
                improved = True

    # Se obtienen todas las métricas
    print(f"THE BEST MSE IS: ", best_MSE)
    errors_mean = me.calculateErrorMean(errors)
    print(f"Average of errors: ", errors_mean)

    if len(errors) > 1:
        error_variance = me.calculateVariance(errors)
        print(f"variance of errors: ", error_variance)
        standard_deviation = me.calculateStandardDesviation(errors)
        print(f"Standard desviation of errors: ", standard_deviation)
    else:
        print("variance of errors: 0.0")
        print("Standard desviation of errors: 0.0")

    return best_breaking_points

# Función del main
if __name__ == '__main__':
    filename, k_segments = me.select_series()

    series_data = me.readSeries(filename)
    breaking_points = me.getBreakingPoints(len(series_data),k_segments)
    best_breaking_points = hillClimbingSearch(me.readSeries(filename),k_segments,breaking_points)
    me.draw(series_data,best_breaking_points, filename)
