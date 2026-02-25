import metrics as me

def hillClimbingSearch(series, k_segments, prev_breaking_points): # Sacar nuevos puntos vecinos que sean mejores que los anteriores
    errors = []
    improved = True 
    neighbour_distance = int(len(series) * 0.005)
    while improved:
        i = 1
        improved = False
        while i < k_segments :
            prev_mean_mse = me.avgMSE(series,prev_breaking_points)
            me.clear_screen()
            print('\n-- HILL CLIMBING SEARCH --')
            print(prev_breaking_points)
            print(f'Average MSE: {me.avgMSE(series,prev_breaking_points)}')
            inc_breaking_points = prev_breaking_points.copy()
            if prev_breaking_points[i] + neighbour_distance < prev_breaking_points[i+1]:
                inc_breaking_points[i] += neighbour_distance
            dec_breaking_points = prev_breaking_points.copy()
            if prev_breaking_points[i] - neighbour_distance > prev_breaking_points[i-1]:
                dec_breaking_points[i] -= neighbour_distance
            mean_mse_inc = me.avgMSE(series,inc_breaking_points)
            mean_mse_dec = me.avgMSE(series,dec_breaking_points)

            if mean_mse_inc < prev_mean_mse and mean_mse_dec >= prev_mean_mse: # Mejora incrementando y empeora decrementando
                prev_breaking_points = inc_breaking_points
                improved = True
                errors.append(mean_mse_inc)

            elif mean_mse_inc >= prev_mean_mse and mean_mse_dec < prev_mean_mse: # Mejora decrementando y empeora incrementando
                prev_breaking_points = dec_breaking_points
                improved = True
                errors.append(mean_mse_dec)

            elif mean_mse_inc < prev_mean_mse and mean_mse_dec < prev_mean_mse: # Mejora en los dos casos
                if mean_mse_inc < mean_mse_dec: #Mejora más incrementando
                    prev_breaking_points = inc_breaking_points
                    improved = True
                    errors.append(mean_mse_inc)
                else: # Mejora más (o igual) decrementando
                    prev_breaking_points = dec_breaking_points
                    improved = True
                    errors.append(mean_mse_dec)
            else: # Empeora en ambos casos
                i += 1
    errors_mean = me.calculateErrorMean(errors)
    print(f"Average of errors: ", errors_mean)
    error_variance = me.calculateVariance(errors)
    print(f"variance of errors: ", error_variance)
    standard_desviation = me.calculateStandardDesviation(errors)
    print(f"Standard desviation of errors: ", standard_desviation)

    return prev_breaking_points

series = me.readSeries('TS1.txt')
k_segments = 9
prev_breaking_points = me.getBreakingPoints(len(series),k_segments)
best_breaking_points = hillClimbingSearch(series,k_segments,prev_breaking_points)
me.draw('TS1.txt', best_breaking_points)
