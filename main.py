import time
import metrics as me
from BusquedaAleatoria import serialRandomSearch, paralelRandomSearch
from HillClimbing import hillClimbingSearch
from simmulatedAnnealing import simmulatedAnnealing

def mostrar_menu_principal():
    me.clear_screen()
    print("==============================")
    print("    ESTUDIO METAHEURÍSTICAS   ")
    print("==============================")
    print("1. Random Search (Serie)")
    print("2. Random Search (Paralelo)")
    print("3. Hill Climbing (Sin límite de iteraciones)")
    print("4. Simulated Annealing")
    print("5. Salir")
    print("==============================")

def ejecutar_estudio_parametrico(algoritmo_func, nombre, series_data, k_segments, filename, repeticiones, it_inicio, it_fin, it_paso):
    
    iteraciones_list = list(range(it_inicio, it_fin + 1, it_paso))
    
    mses_medios_por_iteracion = []
    
    # Nombre del fichero para guardar el resumen
    nombre_seguro = nombre.replace(' ', '_')
    serie_limpia = filename.replace('.txt', '')
    nombre_fichero = f"Estudio_{nombre_seguro}_{serie_limpia}.txt"

    with open(nombre_fichero, "w", encoding="utf-8") as f:
        f.write(f"=========================================\n")
        f.write(f" ESTUDIO DE CONVERGENCIA: {nombre.upper()}\n")
        f.write(f"=========================================\n")
        f.write(f"Fichero: {filename} | K={k_segments}\n")
        f.write(f"Repeticiones por bloque: {repeticiones}\n")
        f.write("-" * 50 + "\n")

        # BUCLE PRINCIPAL: Iteramos sobre los diferentes valores de iteraciones (10, 20, 30...)
        for max_iters in iteraciones_list:
            print(f"\n--> Evaluando con {max_iters} iteraciones máximas...")
            mses_actuales = []
            tiempos_actuales = []
            
            # Hacemos las 3 (o las que sean) pasadas para esta configuración
            for i in range(1, repeticiones + 1):
                print(f"  [Repetición {i}/{repeticiones}]", end="\r")
                start_time = time.time()
                
                # OJO: Los algoritmos ahora reciben las iteraciones por parámetro
                if nombre == "Random Search Serie":
                    puntos_optimos = algoritmo_func(series_data, k_segments, max_iters)
                elif nombre == "Random Search Paralelo":
                    puntos_optimos = algoritmo_func(series_data, k_segments, max_iters, batch=5)
                elif nombre == "Simulated Annealing":
                    puntos_optimos = algoritmo_func(series_data, k_segments, T0=100, L=30, Tf=10.5, max_iter=max_iters)
                else: # Hill Climbing no tiene max_iters
                    puntos_iniciales = me.getBreakingPoints(len(series_data), k_segments)
                    puntos_optimos = algoritmo_func(series_data, k_segments, puntos_iniciales)
                    
                exec_time = time.time() - start_time
                final_mse = me.avgMSE(series_data, puntos_optimos)
                
                mses_actuales.append(final_mse)
                tiempos_actuales.append(exec_time)
            
            # Calculamos estadísticas de este bloque (ej: de las 3 ejecuciones con 10 iteraciones)
            mse_medio = me.calculateErrorMean(mses_actuales)
            tiempo_medio = me.calculateErrorMean(tiempos_actuales)
            var_mse = me.calculateVariance(mses_actuales) if repeticiones > 1 else 0.0
            std_mse = me.calculateStandardDesviation(mses_actuales) if repeticiones > 1 else 0.0
            
            mses_medios_por_iteracion.append(mse_medio)

            # Escribimos en el txt
            f.write(f"\n[ Max Iteraciones: {max_iters} ]\n")
            f.write(f"MSE Medio:           {mse_medio:.6f}\n")
            f.write(f"Varianza MSE:        {var_mse:.6f}\n")
            f.write(f"Desviación MSE:      {std_mse:.6f}\n")
            f.write(f"Tiempo Medio:        {tiempo_medio:.6f} s\n")
    
    print("\n" + "="*50)
    print(f"ESTUDIO TERMINADO. Datos guardados en: {nombre_fichero}")
    print("="*50)
    
    # Dibujamos la regresión solo si hay más de un punto a evaluar
    if len(iteraciones_list) > 1:
        me.draw_iterations_study(iteraciones_list, mses_medios_por_iteracion, nombre, filename)

def main():
    while True:
        mostrar_menu_principal()
        opcion = input("\nSelecciona una opción (1-5): ")

        if opcion == '5':
            print("Saliendo del programa...")
            break

        if opcion in ['1', '2', '3', '4']:
            filename, k_segments = me.select_series()
            series_data = me.readSeries(filename)
            
            # Preguntamos por la configuración del estudio
            if opcion in ['1', '2', '4']:
                print("\n--- CONFIGURACIÓN DEL ESTUDIO ---")
                repeticiones = int(input("¿Cuántas repeticiones para calcular la media? (ej. 3): "))
                it_inicio = int(input("¿Iteración inicial? (ej. 10): "))
                it_fin = int(input("¿Iteración final? (ej. 50): "))
                it_paso = int(input("¿Tamaño del salto/paso? (ej. 10): "))
            else:
                # Para Hill Climbing solo pedimos repeticiones
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
            
            input("\nPresiona Enter para volver al menú...")

if __name__ == "__main__":
    main()