import csv
import numpy as np
import matplotlib.pyplot as plt

ARCHIVO_DATOS = "resultados.csv"
tiempos = []
fitnesses = []

# 1. Leer los datos acumulados
try:
    with open(ARCHIVO_DATOS, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tiempos.append(float(row["Tiempo_s"]))
            fitnesses.append(float(row["Fitness"]))
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {ARCHIVO_DATOS}.")
    print("Ejecuta primero el script de evaluación para generar los datos.")
    exit()

if not tiempos:
    print("El archivo CSV está vacío.")
    exit()

num_ejecuciones = len(tiempos)

# 2. Imprimir el resumen en consola
print("\n=== RESUMEN ESTADÍSTICO ===")
print(f"Ejecuciones leídas: {num_ejecuciones}")
print(f"Tiempo Medio:       {np.mean(tiempos):.2f}s (+/- {np.std(tiempos):.2f}s)")
print(f"Fitness Medio:      {np.mean(fitnesses):.4f} (+/- {np.std(fitnesses):.4f})")
print(f"Mejor Fitness:      {np.min(fitnesses):.4f}")
print("===========================\n")

# 3. Dibujar las gráficas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfica de Tiempos
ax1.boxplot(tiempos, patch_artist=True, boxprops=dict(facecolor='lightblue'))
ax1.scatter([1]*num_ejecuciones, tiempos, color='red', alpha=0.6, zorder=3)
ax1.set_title('Distribución de Tiempo')
ax1.set_ylabel('Segundos')
ax1.set_xticks([1])
ax1.set_xticklabels([f'{num_ejecuciones} runs'])
ax1.grid(True, alpha=0.3)

# Gráfica de Fitness
ax2.boxplot(fitnesses, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
ax2.scatter([1]*num_ejecuciones, fitnesses, color='red', alpha=0.6, zorder=3)
ax2.set_title('Distribución del Fitness')
ax2.set_ylabel('Fitness (Menor es mejor)')
ax2.set_xticks([1])
ax2.set_xticklabels([f'{num_ejecuciones} runs'])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graficas_estadisticas.png', dpi=150)
print("Gráficas guardadas como 'graficas_estadisticas.png'")
plt.show()