import time
import metrics as me
from BusquedaAleatoria import serialRandomSearch, paralelRandomSearch
from HillClimbing import hillClimbingSearch
from simmulatedAnnealing import geometricCooling, simmulatedAnnealing

def mostrar_menu_principal():
    me.clear_screen()
    print("==============================")
    print("    ESTUDIO METAHEURÍSTICAS   ")
    print("==============================")
    print("1. Random Search (Serie)")
    print("2. Random Search (Paralelo)")
    print("3. Hill Climbing (Sin límite de iteraciones)")
    print("4. Simulated Annealing")
    print("5. Comparativa Global de Algoritmos")
    print("6. Salir")
    print("==============================")

def ejecutar_estudio_parametrico(algoritmo_func, nombre, series_data, k_segments, filename, repeticiones, it_inicio, it_fin, it_paso):
    iteraciones_list = list(range(it_inicio, it_fin + 1, it_paso))
    mses_medios_list, varianzas_list, desviaciones_list, tiempos_list = [], [], [], []
    
    nombre_seguro = nombre.replace(' ', '_')
    serie_limpia = filename.replace('.txt', '')
    nombre_fichero = f"./test_files/Estudio_{nombre_seguro}_{serie_limpia}.txt"
    
    best_sol = []
    best_mse = float('inf')

    with open(nombre_fichero, "w", encoding="utf-8") as f:
        f.write(f"=== ESTUDIO DE CONVERGENCIA: {nombre.upper()} ===\n")
        f.write(f"Fichero: {filename} | K={k_segments} | Repeticiones: {repeticiones}\n")
        f.write("-" * 50 + "\n")

        for max_iters in iteraciones_list:
            print(f"\n--> Evaluando con {max_iters} iteraciones máximas...")
            mses_act, tiempos_act = [], []
            
            for i in range(1, repeticiones + 1):
                print(f"  [Repetición {i}/{repeticiones}]", end="\r")
                start_time = time.time()
                
                if nombre == "Random Search Serie":
                    ptos = algoritmo_func(series_data, k_segments, max_iters)
                elif nombre == "Random Search Paralelo":
                    ptos = algoritmo_func(series_data, k_segments, max_iters, batch=5)
                elif nombre == "Simulated Annealing":
                    ptos, _ = algoritmo_func(series_data, k_segments, 100, 30, 0.01, geometricCooling ,max_iters)
                else: 
                    ptos_ini = me.getBreakingPoints(len(series_data), k_segments)
                    ptos = algoritmo_func(series_data, k_segments, ptos_ini)
                    
                exec_time = time.time() - start_time
                mses_act.append(me.avgMSE(series_data, ptos))
                
                if(mses_act[i-1] < best_mse):
                    best_mse = mses_act[i-1]
                    best_sol = ptos
                
                tiempos_act.append(exec_time)
            
            mse_med = me.calculateErrorMean(mses_act)
            tiempo_med = me.calculateErrorMean(tiempos_act)
            var_mse = me.calculateVariance(mses_act) if repeticiones > 1 else 0.0
            std_mse = me.calculateStandardDesviation(mses_act) if repeticiones > 1 else 0.0
            
            mses_medios_list.append(mse_med)
            varianzas_list.append(var_mse)
            desviaciones_list.append(std_mse)
            tiempos_list.append(tiempo_med)

            f.write(f"\n[ Max Iters: {max_iters} ] MSE: {mse_med:.6f} | Var: {var_mse:.6f} | T: {tiempo_med:.6f}s\n")
            f.write(f"\n[Best Solution: {best_sol}]")
            
            me.draw(series_data, best_sol, "Best_Solution")
    
    print("\n" + "="*50)
    print(f"ESTUDIO TERMINADO. Datos guardados en: {nombre_fichero}")
    
    if len(iteraciones_list) > 1:
        stats_dict = {"MSE Medio": mses_medios_list, "Varianza": varianzas_list, "Desviación Típica": desviaciones_list, "Tiempo (s)": tiempos_list}
        while True:
            print("\n--- VISUALIZACIÓN ---")
            print("1. MSE Medio | 2. Varianza | 3. Desviación | 4. Tiempo | 5. Todas (Apiladas) | 6. Salir")
            op = input("Elige (1-6): ")
            if op == '1': me.draw_single_stat_study(iteraciones_list, mses_medios_list, "MSE Medio", nombre, filename)
            elif op == '2': me.draw_single_stat_study(iteraciones_list, varianzas_list, "Varianza", nombre, filename)
            elif op == '3': me.draw_single_stat_study(iteraciones_list, desviaciones_list, "Desviación Típica", nombre, filename)
            elif op == '4': me.draw_single_stat_study(iteraciones_list, tiempos_list, "Tiempo (s)", nombre, filename)
            elif op == '5': me.draw_iterations_study(iteraciones_list, stats_dict, nombre, filename)
            elif op == '6': break

def ejecutar_comparativa_global(series_data, k_segments, filename, repeticiones, it_inicio, it_fin, it_paso):
    iteraciones_list = list(range(it_inicio, it_fin + 1, it_paso))
    
    algoritmos = {
        "Random Search Serie": serialRandomSearch,
        "Random Search Paralelo": paralelRandomSearch,
        "Simulated Annealing": simmulatedAnnealing
    }
    
    comp_mse = {alg: [] for alg in algoritmos}
    comp_var = {alg: [] for alg in algoritmos}
    comp_std = {alg: [] for alg in algoritmos}
    comp_tiempo = {alg: [] for alg in algoritmos}
    
    me.clear_screen()
    print("==================================================")
    print(" INICIANDO COMPARATIVA GLOBAL DE METAHEURÍSTICAS  ")
    print("==================================================")
    
    print("\n--> [1/2] Calculando baseline: Hill Climbing...")
    hc_mses, hc_tiempos = [], []
    for i in range(1, repeticiones + 1):
        start = time.time()
        ptos_ini = me.getBreakingPoints(len(series_data), k_segments)
        ptos = hillClimbingSearch(series_data, k_segments, ptos_ini)
        hc_tiempos.append(time.time() - start)
        hc_mses.append(me.avgMSE(series_data, ptos))
    
    hc_mse_base = me.calculateErrorMean(hc_mses)
    hc_var_base = me.calculateVariance(hc_mses) if repeticiones > 1 else 0.0
    hc_std_base = me.calculateStandardDesviation(hc_mses) if repeticiones > 1 else 0.0
    hc_tiempo_base = me.calculateErrorMean(hc_tiempos)

    print("\n--> [2/2] Evaluando algoritmos iterativos...")
    for max_iters in iteraciones_list:
        print(f"\n  [ Iteraciones: {max_iters} ]")
        
        for nombre, func in algoritmos.items():
            mses_act, tiempos_act = [], []
            for _ in range(repeticiones):
                start = time.time()
                if nombre == "Random Search Serie":
                    ptos = func(series_data, k_segments, max_iters)
                elif nombre == "Random Search Paralelo":
                    ptos = func(series_data, k_segments, max_iters, batch=5)
                elif nombre == "Simulated Annealing":
                    ptos, _ = func(series_data, k_segments, 100, 30, 0.01, geometricCooling, max_iters)
                tiempos_act.append(time.time() - start)
                mses_act.append(me.avgMSE(series_data, ptos))
            
            comp_mse[nombre].append(me.calculateErrorMean(mses_act))
            comp_tiempo[nombre].append(me.calculateErrorMean(tiempos_act))
            comp_var[nombre].append(me.calculateVariance(mses_act) if repeticiones > 1 else 0.0)
            comp_std[nombre].append(me.calculateStandardDesviation(mses_act) if repeticiones > 1 else 0.0)

    print("\n" + "="*50)
    print(" ¡ESTUDIO COMPLETADO CON ÉXITO! ")
    print("="*50)
    
    while True:
        print("\n--- VISUALIZADOR DE COMPARATIVA ---")
        print("1. Comparar MSE Medio")
        print("2. Comparar Varianza del MSE")
        print("3. Comparar Desviación Típica del MSE")
        print("4. Comparar Tiempos de Ejecución")
        print("5. Salir al menú principal")
        
        op = input("\nElige qué gráfica comparativa deseas ver (1-5): ")
        
        if op == '1':
            me.draw_comparison_study(iteraciones_list, comp_mse, "MSE Medio", filename, baseline_val=hc_mse_base)
        elif op == '2':
            me.draw_comparison_study(iteraciones_list, comp_var, "Varianza del MSE", filename, baseline_val=hc_var_base)
        elif op == '3':
            me.draw_comparison_study(iteraciones_list, comp_std, "Desviación Típica del MSE", filename, baseline_val=hc_std_base)
        elif op == '4':
            me.draw_comparison_study(iteraciones_list, comp_tiempo, "Tiempo Medio (s)", filename, baseline_val=hc_tiempo_base)
        elif op == '5':
            break
        else:
            print("Opción no válida.")

def main():
    while True:
        mostrar_menu_principal()
        opcion = input("\nSelecciona una opción (1-6): ")

        if opcion == '6':
            print("Saliendo del programa...")
            break

        if opcion in ['1', '2', '3', '4', '5']:
            filename, k_segments = me.select_series()
            series_data = me.readSeries(filename)
            
            if opcion in ['1', '2', '4', '5']:
                print("\n--- CONFIGURACIÓN DEL ESTUDIO ---")
                repeticiones = int(input("¿Cuántas repeticiones para calcular la media? (ej. 3): "))
                it_inicio = int(input("¿Iteración inicial? (ej. 10): "))
                it_fin = int(input("¿Iteración final? (ej. 50): "))
                it_paso = int(input("¿Tamaño del salto/paso? (ej. 10): "))
            else:
                repeticiones = int(input("¿Cuántas ejecuciones de Hill Climbing deseas hacer? (ej. 5): "))
                it_inicio, it_fin, it_paso = 1, 1, 1

            if opcion == '1':
                ejecutar_estudio_parametrico(serialRandomSearch, "Random Search Serie", series_data, k_segments, filename, repeticiones, it_inicio, it_fin, it_paso)
            elif opcion == '2':
                ejecutar_estudio_parametrico(paralelRandomSearch, "Random Search Paralelo", series_data, k_segments, filename, repeticiones, it_inicio, it_fin, it_paso)
            elif opcion == '3':
                ejecutar_estudio_parametrico(hillClimbingSearch, "Hill Climbing", series_data, k_segments, filename, repeticiones, it_inicio, it_fin, it_paso)
            elif opcion == '4':
                ejecutar_estudio_parametrico(simmulatedAnnealing, "Simulated Annealing", series_data, k_segments, filename, repeticiones, it_inicio, it_fin, it_paso)
            elif opcion == '5':
                ejecutar_comparativa_global(series_data, k_segments, filename, repeticiones, it_inicio, it_fin, it_paso)
            
            input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    main()
