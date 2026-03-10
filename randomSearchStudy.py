import time
import metrics as me
from BusquedaAleatoria import serialRandomSearch, paralelRandomSearch

def mostrar_menu_principal():
    me.clear_screen()
    print("==================================")
    print("   ESTUDIO RANDOM SEARCH (RS)     ")
    print("==================================")
    print("1. Estudio RS Serie (Paramétrico)")
    print("2. Estudio RS Paralelo (Paramétrico)")
    print("3. Búsqueda de Alta Precisión (Hasta el límite)")
    print("4. Comparativa Serie vs Paralelo")
    print("5. Salir")
    print("==================================")

def ejecutar_busqueda_alta_precision(algoritmo_func, nombre, series_data, k_segments, filename):
    """
    Busca de forma indefinida hasta que no hay ninguna mejora en 10,000 iteraciones.
    """
    print(f"\n--- INICIANDO BÚSQUEDA DE ALTA PRECISIÓN ({nombre}) ---")
    print("Configuración: Paciencia = 10,000 iters | Umbral mejora = > 0")
    
    paciencia_max = 10000
    mejor_mse = float('inf')
    iters_sin_mejora = 0
    iteracion_total = 0
    historial_mejores = []
    
    start_time = time.time()
    
    try:
        while iters_sin_mejora < paciencia_max:
            iteracion_total += 1
            
            # Realizamos una búsqueda (batch pequeño para paralelo)
            if "Paralelo" in nombre:
                ptos = algoritmo_func(series_data, k_segments, 1, batch=10)
            else:
                ptos = algoritmo_func(series_data, k_segments, 1)
            
            mse_actual = me.avgMSE(series_data, ptos)
            
            # Si hay CUALQUIER mejora, por mínima que sea
            if mse_actual < mejor_mse:
                mejor_mse = mse_actual
                iters_sin_mejora = 0
                print(f"  [*] ¡Mejora! Iter {iteracion_total}: MSE = {mejor_mse:.6f}          ", end="\r")
            else:
                iters_sin_mejora += 1
            
            historial_mejores.append(mejor_mse)
            
            if iteracion_total % 500 == 0:
                print(f"  [.] Iter: {iteracion_total} | Paciencia: {iters_sin_mejora}/{paciencia_max}          ", end="\r")

    except KeyboardInterrupt:
        print("\n\nBúsqueda detenida manualmente.")

    total_time = time.time() - start_time
    print(f"\n\n¡PROCESO FINALIZADO!")
    print(f"Iteraciones totales: {iteracion_total}")
    print(f"Mejor MSE alcanzado: {mejor_mse:.6f}")
    print(f"Tiempo empleado: {total_time:.2f} segundos")
    
    # Guardar en CSV el resultado final
    me.save_statistics("resultados_rs.csv", nombre + "_MAX", filename, k_segments, iteracion_total, total_time, mejor_mse, 0, 0)
    
    me.draw_convergence_best(historial_mejores, nombre)

def ejecutar_estudio_completo(algoritmo_func, nombre, series_data, k_segments, filename):
    print("\n--- CONFIGURACIÓN DEL EXPERIMENTO ---")
    repeticiones = int(input("¿Repeticiones por cada nivel? (ej. 20): "))
    it_inicio = int(input("¿Inicio? (ej. 10): "))
    it_fin = int(input("¿Fin? (ej. 100): "))
    it_paso = int(input("¿Paso? (ej. 20): "))

    iteraciones_list = list(range(it_inicio, it_fin + 1, it_paso))
    mses_medios, desviaciones, varianzas, tiempos = [], [], [], []
    matriz_para_cuartiles = []

    for max_iters in iteraciones_list:
        mses_actuales, tiempos_actuales = [], []
        print(f"\n[Procesando {max_iters} iters...]", end="")
        for i in range(repeticiones):
            start = time.time()
            if "Paralelo" in nombre:
                ptos = algoritmo_func(series_data, k_segments, max_iters, batch=5)
            else:
                ptos = algoritmo_func(series_data, k_segments, max_iters)
            tiempos_actuales.append(time.time() - start)
            mses_actuales.append(me.avgMSE(series_data, ptos))
        
        m_mse, v_mse, s_mse = me.calculateErrorMean(mses_actuales), me.calculateVariance(mses_actuales), me.calculateStandardDesviation(mses_actuales)
        m_t = me.calculateErrorMean(tiempos_actuales)
        
        mses_medios.append(m_mse); varianzas.append(v_mse); desviaciones.append(s_mse); tiempos.append(m_t)
        matriz_para_cuartiles.append(mses_actuales)
        
        me.save_statistics("resultados_rs.csv", nombre, filename, k_segments, max_iters, m_t, m_mse, v_mse, s_mse)

    while True:
        print(f"\n--- RESULTADOS: {nombre} ---")
        print("1. MSE Medio + Desviación (Área) | 2. Cuartiles por Rango | 3. Gráfica Varianza/Std | 4. Volver")
        op = input("Selecciona: ")
        if op == '1': me.draw_single_stat_with_variance(iteraciones_list, mses_medios, desviaciones, "MSE Medio", nombre, filename)
        elif op == '2': me.analizar_distribucion_cuartiles(matriz_para_cuartiles, nombre, filename)
        elif op == '3': me.draw_variance_std_study(iteraciones_list, varianzas, desviaciones, nombre, filename)
        elif op == '4': break

def main():
    while True:
        mostrar_menu_principal()
        opcion = input("\nElige una opción: ")
        if opcion == '5': break
        
        if opcion in ['1', '2', '3', '4']:
            filename, k_segments = me.select_series()
            series_data = me.readSeries(filename)
            
            if opcion == '1':
                ejecutar_estudio_completo(serialRandomSearch, "RS Serie", series_data, k_segments, filename)
            elif opcion == '2':
                ejecutar_estudio_completo(paralelRandomSearch, "RS Paralelo", series_data, k_segments, filename)
            elif opcion == '3':
                print("1. Serie | 2. Paralelo")
                sub = input("Variante: ")
                func = serialRandomSearch if sub == '1' else paralelRandomSearch
                nom = "RS Serie" if sub == '1' else "RS Paralelo"
                ejecutar_busqueda_alta_precision(func, nom, series_data, k_segments, filename)
            elif opcion == '4':
                print("Lanzando comparativa de distribuciones...")
                # Aquí se puede añadir lógica adicional de comparación si se desea

            input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    main()