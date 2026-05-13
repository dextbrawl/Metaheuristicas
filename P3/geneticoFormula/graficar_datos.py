import csv
import numpy as np
import matplotlib.pyplot as plt

ARCHIVO_DATOS = "resultados.csv"
tiempos = []
fitnesses = []
desviaciones = []

# Leer los datos del CSV
try:
    with open(ARCHIVO_DATOS, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tiempos.append(float(row["Tiempo_s"]))
            fitnesses.append(float(row["Fitness"]))
            desviaciones.append(float(row["Desviacion"]))
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {ARCHIVO_DATOS}.")
    exit()

if not tiempos:
    print("El archivo CSV está vacío.")
    exit()

num_ejecuciones = len(tiempos)
ejecuciones = np.arange(1, num_ejecuciones + 1)

# Imprimir resumen por consola
print("\n=== RESUMEN ESTADÍSTICO ===")
print(f"Ejecuciones leídas: {num_ejecuciones}")
print(f"Tiempo Medio:       {np.mean(tiempos):.2f}s (+/- {np.std(tiempos):.2f}s)")
print(f"Fitness Medio:      {np.mean(fitnesses):.4f} (+/- {np.std(fitnesses):.4f})")
print(f"Desviación Media:   {np.mean(desviaciones):.4f}")
print("===========================\n")

# =========================================================
# 1. GRÁFICA INDEPENDIENTE: TIEMPOS
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(ejecuciones, tiempos, marker='o', linestyle='-', color='dodgerblue', alpha=0.8, label='Tiempo')
plt.axhline(np.mean(tiempos), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(tiempos):.2f}s')
plt.title('Tiempos de Ejecución por Lanzamiento', fontsize=14)
plt.xlabel('Número de Ejecución', fontsize=12)
plt.ylabel('Segundos', fontsize=12)
plt.xticks(ejecuciones) # Para que salgan los números enteros en el eje X
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_01_tiempos.png', dpi=150)
plt.close() # Cerramos la figura para que no se superponga con la siguiente

# =========================================================
# 2. GRÁFICA INDEPENDIENTE: FITNESS
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(ejecuciones, fitnesses, marker='s', linestyle='-', color='forestgreen', alpha=0.8, label='Mejor Fitness')
plt.axhline(np.mean(fitnesses), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(fitnesses):.4f}')
plt.title('Mejor Fitness Alcanzado por Lanzamiento', fontsize=14)
plt.xlabel('Número de Ejecución', fontsize=12)
plt.ylabel('Fitness (Menor es mejor)', fontsize=12)
plt.xticks(ejecuciones)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_02_fitness.png', dpi=150)
plt.close()

# =========================================================
# 3. GRÁFICA INDEPENDIENTE: DESVIACIÓN TÍPICA
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(ejecuciones, desviaciones, marker='^', linestyle='-', color='darkorchid', alpha=0.8, label='Desviación Típica')
plt.axhline(np.mean(desviaciones), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(desviaciones):.1f}')
plt.title('Desviación Típica Final por Lanzamiento', fontsize=14)
plt.xlabel('Número de Ejecución', fontsize=12)
plt.ylabel('Desviación (Escala Logarítmica)', fontsize=12)
plt.yscale('log') # Clave para que los picos gigantes no estropeen la gráfica
plt.xticks(ejecuciones)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_03_desviacion.png', dpi=150)
plt.close()

print("[+] Las gráficas se han generado de forma independiente:")
print("  - grafica_01_tiempos.png")
print("  - grafica_02_fitness.png")
print("  - grafica_03_desviacion.png")