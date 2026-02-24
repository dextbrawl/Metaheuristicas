import subprocess

def mostrar_menu():
    """Imprime las opciones del menú en la pantalla."""
    print("\n" + "="*25)
    print("      MENÚ PRINCIPAL      ")
    print("="*25)
    print("1. RandomSearch")
    print("2. HillClimbing")
    print("3. Simulated Annealing")
    print("4. Salir")
    print("="*25)

def random_search():
    while True:
        print("\n--> ¿Cómo quieres hacer el estudio?")
        print("1. Serie")
        print("2. Paralelo")
        print("3. Salir")
        opcion = input("\nSelecciona una opción (1-3): ")

        if opcion == '1':
            print(f"Llamada a random_search en serie")
            subprocess.run(["python"], "BusquedaAleatoria.py")
        elif opcion == '2':
            print(f"Llamada a random_search en paralelo")
        elif opcion == '3':
            print("\nVolviendo al menú principal... ¡Nos vemos!")
            break
        else:
            print("\n Opción no válida. Por favor, introduce un número del 1 al 3.")

def hill_climbing():
    print("\n--> Ejecutando el método Hill Climbing.")
    subprocess.run(["python"], "HillClimbing.py")

def accion_tres():
    print("\n--> Ejecutando el método Simulated Annealing.")
    # subprocess.run(["python"], "")

def main():
    while True:
        mostrar_menu()
        opcion = input("\nSelecciona una opción (1-4): ")

        if opcion == '1':
            random_search()
        elif opcion == '2':
            accion_dos()
        elif opcion == '3':
            accion_tres()
        elif opcion == '4':
            print("\nSaliendo del programa... ¡Hasta luego!")
            break
        else:
            print("\n Opción no válida. Por favor, introduce un número del 1 al 4.")

if __name__ == "__main__":
    main()