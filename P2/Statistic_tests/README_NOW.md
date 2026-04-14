# COMO HACER LAS ESTADISTICAS
nos ajustamos al número de combinaciones posibles del gridsearch.
cosas a sacar de gridsearch y RS:

1. media de los accuracys
2. varianza y desviación típica de las soluciones, del total.
3. tiempo ejecución
4. CPU
5. Memoria
6. mejor accuracy

Para el genético:

1. Mismo número de individuos generados totales que los anteriores.
2. Tamaño de población ---> 50
3. número de iteraciones ---> 350
4. Estudio generacional de la siguiente manera:
  1. mejor accuracy de la población, de la primera, y luego de cada población final(tras reemplazamiento)
  2. Accuracy medio de la población(del accuracy), de la primera, y luego de cada población final(tras reemplazamiento)
  3. varianza y desviación típica de la población(del accuracy), de la primera, y luego de cada población final(tras reemplazamiento)
  4. tiempo(total del algoritmo)
  5. CPU(total del algoritmo)
  6. Memoria(total del algoritmo)
    
  # A PODER SER SACAR LOS RESULTADOS TMB DE CADA ITERACIÓN (del accuracy y demás) PARA LOS TESTS ESTADÍSTICOS
