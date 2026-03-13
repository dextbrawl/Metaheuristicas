import HillClimbing as hc
import sys
import os

# Para poder importar el módulo desde carpetas externas
aux_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'aux'))
sys.path.append(aux_path)
import metrics as me
import time
import csv

series_dict = {
    "TS1.txt": 9,"TS2.txt": 10,"TS3.txt": 20,"TS4.txt": 50
}

def generateCSV(n_repetitions):

    data = []
    data.append(['Número Serie','Media Error','Mejor Error','Varianza Error','Media Tiempo (s)'])

    i = 1

    for filename, k_segments in series_dict.items():
        errors = []
        times = []
        
        series = me.readSeries(filename)
        best_breaking_points = me.getBreakingPoints(len(series),k_segments)
        best_mse = me.avgMSE(series,best_breaking_points)
        for _ in range(n_repetitions):
            new_breaking_points = me.getBreakingPoints(len(series),k_segments)
            start_time = time.time()
            new_breaking_points = hc.hillClimbingSearch(series,k_segments,new_breaking_points)
            exec_time = time.time() - start_time

            new_mse = me.avgMSE(series,new_breaking_points)
            
            errors.append(new_mse)
            times.append(exec_time)

            if new_mse < best_mse:
                best_breaking_points = new_breaking_points.copy()
                best_mse = new_mse

        mse_mean = me.calculateErrorMean(errors)
        mse_variance = me.calculateVariance(errors)
        avg_time = me.calculateErrorMean(times)
                
        new_line = []
        
        new_line.append(i)
        new_line.append(mse_mean)
        new_line.append(best_mse)
        new_line.append(mse_variance)
        new_line.append(avg_time)

        data.append(new_line)
        
        me.draw(series,best_breaking_points, filename)

        i += 1


    with open('HCOutput/hillClimbing.csv','w', newline='', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    
generateCSV(2)
