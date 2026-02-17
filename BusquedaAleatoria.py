import random

NPuntos = int(input("Introduce numero de puntos de la serie: "))
KSegmentos = int(input("Introduce numero K de segmentos: "))

print("Empezamos búsqueda aleatoria con", NPuntos, "puntos y", KSegmentos, "segmentos\n")

# Creamos un vector con KSegmentos-1 elementos aleatorios entre 0 y NPuntos
vector = [random.randint(0, NPuntos) for _ in range(KSegmentos - 1)]

#Imprimir (no hace falta pero es para visualizarlo)
for i, val in enumerate(vector):
    print(f"vector[{i}] = {val}")
