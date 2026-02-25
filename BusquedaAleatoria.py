import metrics as me

#Heuristica aleatoria
def randomSearch(series: list, k_segments):
    max_iterations = int(input("Introduce el número máximo de iteraciones: "))
    me.clear_screen()
    print(" -- RANDOM SEARCH --")
    size = len(series)
    
    breaking_points = me.getBreakingPoints(size,k_segments)
    avg_mse = me.avgMSE(series,breaking_points)
    
    print("SEGMENTS")
    print("Average MSE: ", avg_mse) 
    c = 0
    errors = []
    errors.append(avg_mse)
    for _ in range(max_iterations):
        new_breaking_points = me.getBreakingPoints(size, k_segments)
        new_avg_mse = me.avgMSE(series,new_breaking_points)

        if new_avg_mse < avg_mse :
            c = 0
            breaking_points = new_breaking_points
            avg_mse = new_avg_mse
            me.clear_screen()
            print(" -- RANDOM SEARCH --")
            print("Average MSE: ", avg_mse)
            errors.append(avg_mse)

        if c == (max_iterations/2):
            me.clear_screen()
            print("STOP:Too may interactions without improving.")
            print(" -- RANDOM SEARCH --")
            print("Average MSE: ", avg_mse)
            return breaking_points



    #Hago return de los puntos para usarlos en draw
    c = c + 1
    errors_mean = me.calculateErrorMean(errors)
    print(f"Average of errors: ", errors_mean)
    error_variance = me.calculateVariance(errors)
    print(f"variance of errors: ", error_variance)
    standard_desviation = me.calculateStandardDesviation(errors)
    print(f"Standard desviation of errors: ", standard_desviation)
    return breaking_points

