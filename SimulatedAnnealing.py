import metrics as me
import random
import math

def get_random_neighbour(breaking_points, size, percentage=0.01):
    new_points = breaking_points.copy()
    
    neighbour_distance = max(1, int(size * percentage))
    idx = random.randint(1, len(new_points) - 2)
    
    move = random.randint(-neighbour_distance, neighbour_distance)
    new_val = new_points[idx] + move
    
    if new_points[idx-1] < new_val < new_points[idx+1]:
        new_points[idx] = new_val
        
    return new_points

def simulatedAnnealingSearch(series, k_segments):
    T0 = 100.0
    alpha = 0.95
    L = 50
    Tf = 0.01
    
    size = len(series)
    neighbour_distance = max(1, int(size * 0.01))
    
    curr_points = me.getBreakingPoints(size, k_segments)
    curr_mse = me.avgMSE(series, curr_points)
    
    best_points = curr_points.copy()
    best_mse = curr_mse
    
    T = T0
    errors = [curr_mse]

    print(" -- SIMULATED ANNEALING --")
    
    while T >= Tf:
        for _ in range(L):
            cand_points = get_random_neighbour(curr_points, size, neighbour_distance)
            cand_mse = me.avgMSE(series, cand_points)
            
            delta = cand_mse - curr_mse

            if delta < 0 or random.random() < math.exp(-delta / T):
                curr_points = cand_points
                curr_mse = cand_mse
                errors.append(curr_mse)
                
                if curr_mse < best_mse:
                    best_mse = curr_mse
                    best_points = curr_points.copy()

        T *= alpha
        print(f"Temp: {T:.4f} | Best MSE: {best_mse:.6f}", end='\r')

    me.clear_screen()
    print("\n-- SIMULATED ANNEALING FINISHED --")
    print(f"Final Best MSE: {best_mse}")
    
    print(f"Average of errors: {me.calculateErrorMean(errors)}")
    print(f"Standard deviation: {me.calculateStandardDesviation(errors)}")
    
    return best_points

if __name__ == '__main__':
    filename, k_segments = me.select_series()
    series_data = me.readSeries(filename)
    best_bp = simulatedAnnealingSearch(series_data, k_segments)
    me.draw(series_data, best_bp, filename)