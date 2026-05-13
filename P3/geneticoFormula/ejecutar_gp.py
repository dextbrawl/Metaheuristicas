import subprocess
import time
import csv
import os
import sys

ARCHIVO_GP = "GP.py"
ARCHIVO_SALIDA = "resultados.csv"

print(f"Ejecutando {ARCHIVO_GP}...")
inicio = time.time()

# El flag "-u" fuerza a Python a imprimir la salida estándar sin buffer
proceso = subprocess.Popen(
    ["python", "-u", ARCHIVO_GP], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True
)

fitness = None

# Leer y mostrar línea por línea en la consola
for linea in proceso.stdout:
    sys.stdout.write(linea)
    sys.stdout.flush()
    
    if "FITNESS_FINAL:" in linea:
        try:
            fitness = float(linea.strip().split("FITNESS_FINAL:")[1])
        except ValueError:
            pass

proceso.wait()
tiempo_total = time.time() - inicio

print("\n" + "="*40)

if fitness is not None:
    archivo_existe = os.path.isfile(ARCHIVO_SALIDA)
    
    with open(ARCHIVO_SALIDA, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not archivo_existe:
            writer.writerow(["Tiempo_s", "Fitness"])
        writer.writerow([tiempo_total, fitness])
        
    print(f"Proceso finalizado. Tiempo: {tiempo_total:.2f}s | Fitness: {fitness:.4f}")
    print(f"Dato guardado en {ARCHIVO_SALIDA}")
else:
    print("ERROR: No se encontró el texto 'FITNESS_FINAL:' en la salida estándar.")