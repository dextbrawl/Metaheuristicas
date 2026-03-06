import time
import metrics as me
from BusquedaAleatoria import serialRandomSearch, paralelRandomSearch
from HillClimbing import hillClimbingSearch

def mostrar_menu_principal():
    me.clear_screen()
    print("==============================")
    print("    ESTUDIO METAHEURÍSTICAS   ")
    print("==============================")
    print("1. Random Search (Serie)")
    print("2. Random Search (Paralelo)")
    print("3. Hill Climbing")
    print("4. Simulated Annealing (Próximamente)")
    print("5. Salir")
    print("==============================")

def ejecutar_algoritmo(algoritmo_func, nombre, series_data, k_segments, filename, **kwargs):
    print(f"\n--> Ejecutando {nombre}...")
    
    # Medición de tiempo
    start_time = time.time()
    
    # Ejecución (Pasamos los argumentos necesarios según el algoritmo)
    if nombre == "Hill Climbing":
        # Hill Climbing necesita puntos iniciales
        puntos_iniciales = me.getBreakingPoints(len(series_data), k_segments)
        puntos_optimos = algoritmo_func(series_data, k_segments, puntos_iniciales)
    else:
        puntos_optimos = algoritmo_func(series_data, k_segments)
    
    end_time = time.time()
    tiempo_total = end_time - start_time

    print(f"\n[RESULTADOS {nombre.upper()}]")
    print(f"Tiempo de ejecución: {tiempo_total:.4f} segundos")
    
    # Llamada a la gráfica (Asegúrate de que metrics.draw reciba la serie, no el nombre)
    print("Generando gráfica...")
    me.draw(series_data, puntos_optimos, filename, title=f"{nombre} - Tiempo: {tiempo_total:.2f}s")

def main():
    while True:
        mostrar_menu_principal()
        opcion = input("\nSelecciona una opción (1-5): ")

        if opcion == '5':
            print("Saliendo del programa...")
            break

        if opcion in ['1', '2', '3', '4']:
            if opcion == '4':
                print("\n[!] Simulated Annealing aún no está implementado.")
                time.sleep(2)
                continue

            # 1. Seleccionar la serie y cargar datos una sola vez
            filename, k_segments = me.select_series()
            series_data = me.readSeries(filename)

            # 2. Ejecutar según la opción
            if opcion == '1':
                ejecutar_algoritmo(serialRandomSearch, "Random Search Serie", series_data, k_segments, filename)
            elif opcion == '2':
                ejecutar_algoritmo(paralelRandomSearch, "Random Search Paralelo", series_data, k_segments, filename)
            elif opcion == '3':
                ejecutar_algoritmo(hillClimbingSearch, "Hill Climbing", series_data, k_segments, filename)
            
            input("\nPresiona Enter para volver al menú...")

if __name__ == "__main__":
    main()