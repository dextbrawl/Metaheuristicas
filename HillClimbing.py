import metrics as me
import random 
def neighbourhood(breaking_points, step):
    neighbourhood = []
    for i in range(1,len(breaking_points) - 1):
        if breaking_points[i] < breaking_points[i+1] - step:
            new_breaking_points_up = breaking_points.copy()
            new_breaking_points_up[i] += random.randint(1,step)
            neighbourhood.append(new_breaking_points_up)
        if breaking_points[i] > breaking_points[i-1] + step:
            new_breaking_points_down = breaking_points.copy()
            new_breaking_points_down[i] -= random.randint(1,step)
            neighbourhood.append(new_breaking_points_down)
    return neighbourhood

def hillClimbingSearch(series, k_segments, prev_breaking_points): # Sacar nuevos puntos vecinos que sean mejores que los anteriores
    errors = []
    improved = True
    step = 1
    best_breaking_points = prev_breaking_points.copy()
    best_MSE =me.avgMSE(series,prev_breaking_points)
    print(prev_breaking_points)
    while improved:
        nbh = neighbourhood(best_breaking_points,step)
        improved = False
        for a in nbh:
            print(a)
            curr_MSE = me.avgMSE(series,a)
            if curr_MSE < best_MSE:
                best_breaking_points = a.copy()
                best_MSE = curr_MSE
                improved = True

    # errors_mean = me.calculateErrorMean(errors)
    # print(f"Average of errors: ", errors_mean)
    # error_variance = me.calculateVariance(errors)
    # print(f"variance of errors: ", error_variance)
    # standard_deviation = me.calculateStandardDesviation(errors)
    # print(f"Standard deviation of errors: ", standard_deviation)
    print("MSE: ", me.avgMSE(series,best_breaking_points))
    return prev_breaking_points

if __name__ == '__main__':
    filename, k_segments = me.select_series()

    series_data = me.readSeries(filename)
    breaking_points = me.getBreakingPoints(len(series_data),k_segments)
    best_breaking_points = hillClimbingSearch(me.readSeries(filename),k_segments,breaking_points)
    me.draw(series_data,best_breaking_points, filename)
