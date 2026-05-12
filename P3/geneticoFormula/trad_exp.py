import sympy as sp

# 1. Definimos nuestros símbolos matemáticos
x, y = sp.symbols("x y")


# 2. Recreamos el diccionario de DEAP en Python nativo para SymPy
def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def secureDiv(a, b):
    # Para la simplificación algebraica pura, asumimos división normal
    return a / b


# 3. Pegamos tu monstruo como un string
arbol_deap = "add(mul(mul(mul(y, 0.37867730196570193), -1.1550619134517426), y), add(0.20154551265891563, mul(mul(mul(-1.1550619134517426, x), add(mul(mul(y, y), mul(mul(x, mul(x, x)), mul(mul(8.344781739240489, x), x))), x)), 0.37867730196570193)))"
# 4. Evaluamos el string (se ejecutará usando las funciones que acabamos de crear)
# Usamos un bloque try-except por si la simplificación choca con algún 0/0 algebraico
try:
    ecuacion_cruda = eval(arbol_deap)
    print("--- ECUACIÓN COMPLETA ---")
    print(ecuacion_cruda)

    print("\n--- ECUACIÓN SIMPLIFICADA (Lista para GeoGebra) ---")
    # sp.simplify() aplica álgebra pesada para colapsar y cancelar términos
    ecuacion_limpia = sp.simplify(ecuacion_cruda)
    print(f"{ecuacion_limpia} = 0")

except Exception as e:
    print(f"Error al simplificar algebraicamente: {e}")
