import HillClimbing as hc
import metrics as me
import time
import csv

series_dict = {
    "TS1.txt": 9,"TS2.txt": 10,"TS3.txt": 20,"TS4.txt": 50
}

def generateCSV(n_repetitions):

    data = []
    data.append(['Número Serie','Media Error','Varianza Error','Media Tiempo'])

    i = 1

    for filename, k_segments in series_dict.items():
        errors = []
        times = []
        for _ in range(n_repetitions):
            series = me.readSeries(filename)
            starting_breaking_points = me.getBreakingPoints(len(series),k_segments)
            start_time = time.time()
            best_breaking_points = hc.hillClimbingSearch(series,k_segments,starting_breaking_points)
            exec_time = time.time() - start_time 
            mse = me.avgMSE(series,best_breaking_points)
            errors.append(mse)
            times.append(exec_time)
        
        mse_mean = me.calculateErrorMean(errors)
        mse_variance = me.calculateVariance(errors)
        avg_time = me.calculateErrorMean(times)
        
        new_line = []
        
        new_line.append(i)
        new_line.append(mse_mean)
        new_line.append(mse_variance)
        new_line.append(avg_time)

        data.append(new_line)
        
        i += 1

    with open('hillClimbing.csv','w', newline='', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)


generateCSV(20)
