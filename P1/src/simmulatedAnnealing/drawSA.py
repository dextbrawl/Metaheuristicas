import metrics as me
import simmulatedAnnealing as sa

# 1. Pon aquí la serie y la configuración que viste que ganó en tu CSV
filename = "TS4.txt"
k_segments = 9
T0 = 10.0
Tf = 0.001
factor_L = 5.0
coolingFunction = sa.geometricCooling

series_data = me.readSeries(filename)
L_real = int(factor_L * (2 * k_segments))
max_iter = 200

print(f"Buscando una solución óptima para {filename}...")

mejor_mse = float('inf')
mejores_puntos = []

# Ejecutamos unas cuantas veces la mejor configuración
for i in range(10):
    ptos, mse, _ = sa.simmulatedAnnealing(series_data, k_segments, T0, L_real, Tf, coolingFunction, max_iter)
    if mse < mejor_mse:
        mejor_mse = mse
        mejores_puntos = ptos

print(f"¡Solución encontrada! MSE: {mejor_mse}")
print(f"Los puntos de corte son: {mejores_puntos}")

# Graficamos la solución
me.draw(series_data, mejores_puntos, filename, title=f"Mejor SA - {filename} (MSE: {mejor_mse:.4f})")
