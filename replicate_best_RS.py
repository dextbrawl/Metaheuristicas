import time
import metrics as me
from BusquedaAleatoria import paralelRandomSearch

if __name__ == "__main__":
    # Aquí definimos las combinaciones ganadoras (Serie, K, Algoritmo, Iteraciones)
    configs_ganadoras = [
        ("TS1.txt", 9, paralelRandomSearch, "RS_Paralelo", 200),
        ("TS2.txt", 15, paralelRandomSearch, "RS_Paralelo", 150),
        ("TS3.txt", 25, paralelRandomSearch, "RS_Paralelo", 200),
        ("TS4.txt", 50, paralelRandomSearch, "RS_Paralelo", 150)
    ]

    # Le ponemos 50 repeticiones para darle más oportunidades de encontrar 
    # un MSE realmente bajo y que la gráfica quede bonita.
    repeticiones = 50

    print(f"=========================================================")
    print(f" RECUPERANDO LAS MEJORES GRÁFICAS DEL RANDOM SEARCH      ")
    print(f"=========================================================\n")

    for filename, k_segments, func, alg_name, max_iter in configs_ganadoras:
        series_data = me.readSeries(filename)
        print(f">>> Buscando solución top para: {filename} ({alg_name} - {max_iter} iter) <<<")
        
        mejor_mse_global = float('inf')
        mejores_ptos_global = []

        for i in range(repeticiones):
            # 1. Ejecutamos el algoritmo
            ptos = func(series_data, k_segments, max_iter)
            
            # 2. Calculamos el MSE de esos puntos
            mse = me.avgMSE(series_data, ptos)

            # 3. Guardamos si es el mejor que hemos visto
            if mse < mejor_mse_global:
                mejor_mse_global = mse
                mejores_ptos_global = list(ptos)

        print(f" => ¡Mejor solución encontrada! MSE: {mejor_mse_global:.4f}")
        print(f" Puntos de corte: {mejores_ptos_global}\n")
        
        # 4. Dibujamos la gráfica
        titulo_grafica = f"Mejor {alg_name} ({max_iter} iter) - {filename} (MSE: {mejor_mse_global:.4f})"
        me.draw(series_data, mejores_ptos_global, filename, title=titulo_grafica)

    print(f"=========================================================")
    print(f" ¡TODAS LAS GRÁFICAS HAN SIDO GENERADAS!                 ")
    print(f"=========================================================")
