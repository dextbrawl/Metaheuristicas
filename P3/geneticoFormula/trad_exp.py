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
arbol_deap = "add(secureDiv(y, add(8.233279434076152, y)), secureDiv(mul(secureDiv(secureDiv(1.9719960348793997, 9.807691928941708), sub(x, secureDiv(y, add(sub(sub(y, y), sub(sub(secureDiv(x, add(mul(sub(y, mul(sub(sub(secureDiv(mul(secureDiv(secureDiv(1.9719960348793997, 9.807691928941708), sub(x, y)), sub(x, y)), add(mul(sub(y, mul(-6.148082104818324, sub(sub(y, secureDiv(x, add(secureDiv(y, secureDiv(mul(secureDiv(secureDiv(1.9719960348793997, 9.807691928941708), sub(secureDiv(sub(add(x, mul(sub(6.308944228200183, y), sub(-5.093552906183067, y))), x), x), secureDiv(y, add(sub(sub(y, mul(add(y, 2.0187382145216155), add(-1.3691222658834121, y))), y), sub(-6.647546511150868, 6.308944228200183))))), y), add(sub(x, y), -6.647546511150868))), sub(x, x)))), sub(add(mul(x, x), y), sub(x, y))))), mul(6.308944228200183, x)), y)), 6.308944228200183), sub(x, y)), sub(sub(y, y), sub(add(7.0987750269723975, add(y, y)), sub(x, y))))), mul(6.308944228200183, secureDiv(mul(secureDiv(secureDiv(1.9719960348793997, 9.807691928941708), sub(x, secureDiv(y, add(sub(sub(y, y), sub(sub(secureDiv(mul(secureDiv(secureDiv(1.9719960348793997, 9.807691928941708), sub(x, y)), y), add(mul(sub(y, mul(secureDiv(-3.162795406447434, y), sub(sub(y, secureDiv(x, y)), sub(add(7.0987750269723975, y), sub(x, y))))), mul(6.308944228200183, x)), y)), 6.308944228200183), sub(x, y))), sub(secureDiv(mul(secureDiv(secureDiv(1.9719960348793997, x), sub(x, y)), add(y, -0.026484831792208396)), add(mul(sub(y, mul(-6.148082104818324, y)), mul(6.308944228200183, secureDiv(x, y))), y)), 6.308944228200183))))), y), add(mul(sub(sub(secureDiv(sub(add(x, mul(sub(y, y), sub(-5.093552906183067, y))), x), x), mul(y, x)), sub(-6.564574767179023, add(-1.3691222658834121, y))), mul(y, x)), mul(add(y, x), add(-9.122947676461084, y)))))), y)), 6.308944228200183), sub(x, y))), sub(secureDiv(mul(secureDiv(secureDiv(1.9719960348793997, 9.807691928941708), sub(x, y)), add(sub(add(7.0987750269723975, y), mul(add(y, -5.093552906183067), mul(sub(x, -2.8314767417406017), add(y, y)))), -0.026484831792208396)), add(mul(sub(y, mul(-6.148082104818324, y)), mul(6.308944228200183, x)), y)), 6.308944228200183))))), y), add(mul(sub(sub(secureDiv(sub(add(x, mul(sub(y, y), sub(y, y))), x), x), mul(y, x)), sub(-6.564574767179023, add(-1.3691222658834121, 5.658899413011598))), mul(y, x)), mul(add(y, x), add(-9.122947676461084, y)))))"

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
