# Guía de uso para la regresión simbólica

Este documento detalla cómo configurar y ejecutar el script para obtener las fronteras de los modelos A y B. Es fundamental ajustar los parámetros según el modelo que se vaya a evaluar para que la bisección y la evolución funcionen correctamente.

## Consideraciones sobre los parámetros

La configuración varía según el tamaño y la complejidad de la frontera de cada modelo. Para el modelo A se han podido usar un número de puntos y una distancia Inter Par mayor puesto que la frontera que genera parece ser mayor. Sin embargo, con el modelo B, debido a su mayor simpleza y menor tamaño de la frontera, ha debido usarse una menor distancia interpar y un menor número de puntos.

Para evaluar cada caso, debes localizar las variables en el código y modificar sus valores según esta tabla:

### Configuración Modelo A
* Archivo: `blackbox_modelA.pkl`
* `n_points` = 35
* `th_interp_distance` = 0.3
* Generaciones (`ngen`) = 1000

### Configuración Modelo B
* Archivo: `blackbox_modelB.pkl`
* `n_points` = 15
* `th_interp_distance` = 0.15
* Generaciones (`ngen`) = 1000

## Pasos para la ejecución

1. Carga del modelo: Cambia la ruta en la inicialización de `BlackBoxModel` para apuntar al archivo `.pkl` del modelo correspondiente.
2. Ajuste de bisección: Modifica `n_points` y `th_interp_distance` con los valores indicados en la sección anterior. No uses valores superiores en el Modelo B o el bucle de búsqueda podría quedarse bloqueado buscando espacio que no existe.
3. Generaciones: Verifica que el parámetro `ngen` dentro de la llamada a `algorithms.eaSimple` esté configurado siempre en 1000.
4. Ejecución: Lanza el script desde la terminal. Primero verás la extracción de puntos y después la tabla de evolución del algoritmo genético.

## Lectura de estadísticas y salida

Durante la ejecución, la consola mostrará una tabla con varias columnas. La métrica principal es "min", que indica el mejor fitness (error mínimo) de esa generación. El objetivo es que este valor se acerque a cero progresivamente.

Al finalizar las 1000 generaciones:
1. Se imprimirá por consola la mejor ecuación encontrada.
2. Se abrirá una ventana con el renderizado gráfico. La línea roja (la función del algoritmo) debe ajustarse sobre los puntos azules (la frontera de bisección).
3. Anota el valor exacto de `FITNESS_FINAL` que aparece al final del proceso, lo necesitaremos para la memoria comparativa de la práctica.
