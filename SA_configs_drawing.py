import time
import metrics as me
import simmulatedAnnealing as sa

if __name__ == "__main__":
    configs = [
        ("TS1.txt", 9, 1.0, 0.1, 36),
        ("TS2.txt", 15, 1.0, 0.001, 40),
        ("TS3.txt", 25, 1.0, 0.001, 20),
        ("TS4.txt", 50, 50.0, 0.001, 50)
    ]

    repeticiones = 20
    max_iter = 200

    print(f"=========================================================")
    print(f" OBTENIENDO MEJORES GRÁFICAS DEL SA BALANCEADO           ")
    print(f"=========================================================\n")

    for filename, k_segments, T0, Tf, L in configs:
        series_data = me.readSeries(filename)
        print(f">>> Procesando serie: {filename} (Config: T0={T0}, Tf={Tf}, L={L}) <<<")
        
        mejor_mse_global = float('inf')
        mejores_ptos_global = []

        for i in range(repeticiones):
            start_time = time.time()
            
            ptos, mse, _ = sa.simmulatedAnnealing(
                series_data, k_segments, T0, L, Tf, sa.geometricCooling, max_iter
            )
            
            time_elapsed = time.time() - start_time

            if mse < mejor_mse_global:
                mejor_mse_global = mse
                mejores_ptos_global = list(ptos)

            print(f"Rep {i+1:02d} | MSE: {mse:.4f} | Tiempo: {time_elapsed:.4f}s")

        print(f" => ¡Mejor resultado en {filename}! MSE: {mejor_mse_global:.4f}\n")
        
        titulo_grafica = f"Mejor SA Balanceado - {filename} (MSE: {mejor_mse_global:.4f})"
        me.draw(series_data, mejores_ptos_global, filename, title=titulo_grafica)

    print(f"=========================================================")
    print(f" ¡TODAS LAS GRÁFICAS HAN SIDO GENERADAS!                 ")
    print(f"=========================================================")
