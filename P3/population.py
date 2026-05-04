import numpy as np
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import prueba as modelo

#Un individuo es un conjunto de puntos (de pares de puntos)
class Individual:
    
    def __init__(self, num_pairs: int = 15, bounds: Tuple[float, float] = (-50.0, 50.0), model: modelo.BlackBoxModel = None):
        self.num_pairs = num_pairs
        self.bounds = bounds
        self.model = model
        
        if model is not None:
            self.pairing()
        else:
            self.points_class0 = np.random.uniform(bounds[0], bounds[1], (num_pairs, 2))
            self.points_class1 = np.random.uniform(bounds[0], bounds[1], (num_pairs, 2))
            self.all_points = np.vstack([self.points_class0, self.points_class1])
            self.classes = None
            self.fitness = None
    
    def pairing(self):
       
        self.points_class0 = []
        self.points_class1 = []
        
        max_attempts = self.num_pairs * 200  # Aumentamos intentos
        attempts = 0
        
        while len(self.points_class0) < self.num_pairs and attempts < max_attempts:
            point_a = np.random.uniform(self.bounds[0], self.bounds[1], 2)
            point_b = np.random.uniform(self.bounds[0], self.bounds[1], 2)
            
            class_a = self.model.predict(point_a)
            class_b = self.model.predict(point_b)
            
            if class_a != class_b:
                if class_a == 0:
                    self.points_class0.append(point_a)
                    self.points_class1.append(point_b)
                else:
                    self.points_class0.append(point_b)
                    self.points_class1.append(point_a)
            
            attempts += 1
        
        # Si no se generaron suficientes pares, completar con puntos aleatorios
        # y luego ajustar sus clases
        while len(self.points_class0) < self.num_pairs:
            # Generar punto aleatorio
            point = np.random.uniform(self.bounds[0], self.bounds[1], 2)
            point_class = self.model.predict(point)
            
            # Buscar un punto de clase opuesta
            max_attempts_2 = 100
            for _ in range(max_attempts_2):
                other_point = np.random.uniform(self.bounds[0], self.bounds[1], 2)
                other_class = self.model.predict(other_point)
                
                if other_class != point_class:
                    if point_class == 0:
                        self.points_class0.append(point)
                        self.points_class1.append(other_point)
                    else:
                        self.points_class0.append(other_point)
                        self.points_class1.append(point)
                    break
        
        self.points_class0 = np.array(self.points_class0[:self.num_pairs])
        self.points_class1 = np.array(self.points_class1[:self.num_pairs])
        self.all_points = np.vstack([self.points_class0, self.points_class1])
        
        print(f"Generados {len(self.points_class0)}/{self.num_pairs} pares válidos")
    
    def getClasses(self, model: modelo.BlackBoxModel) -> np.ndarray:
        self.classes = np.array([model.predict(point) for point in self.all_points])
        return self.classes
    
    def getDistances(self) -> float:
        if self.classes is None:
            if self.model:
                self.getClasses(self.model)
            else:
                raise ValueError("primero clases bro")
        
        distances = []
        for i in range(self.num_pairs):
            dist = np.linalg.norm(self.points_class0[i] - self.points_class1[i])
            distances.append(dist)
        
        return np.mean(distances)
    
    def getDispersion(self) -> float:
        if len(self.all_points) < 2:
            return 0.0
        
        min_distance = float('inf')
        
        for i in range(self.num_pairs):
            for j in range(i + 1, self.num_pairs):
                d00 = np.linalg.norm(self.points_class0[i] - self.points_class0[j])
                d11 = np.linalg.norm(self.points_class1[i] - self.points_class1[j])
                d01 = np.linalg.norm(self.points_class0[i] - self.points_class1[j])
                d10 = np.linalg.norm(self.points_class1[i] - self.points_class0[j])
                
                for d in [d00, d11, d01, d10]:
                    if d < min_distance:
                        min_distance = d
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def compute_fitness(self, model: modelo.BlackBoxModel) -> float:
       
        #Distancia pares - dispersion
        
        self.getClasses(model)
        dist_pairs = self.getDistances()
        dispersion = self.getDispersion()
        self.fitness = dist_pairs - dispersion
        return self.fitness
    






# Prueba rápida
if __name__ == "__main__":
    # Cargar modelo
    model = modelo.BlackBoxModel("blackbox_modelB.pkl")
    
    print("Creando individuo con pares válidos...")
    ind = Individual(num_pairs=10, model=model)
    
    print(f"\nIndividuo creado con {ind.num_pairs} pares")
    print(f"Puntos clase0: {ind.points_class0.shape}")
    print(f"Puntos clase1: {ind.points_class1.shape}")
    
    # Mostrar puntos
    print("\n=== PUNTOS CLASE 0 ===")
    for i, point in enumerate(ind.points_class0):
        print(f"Punto {i+1}: ({point[0]:.4f}, {point[1]:.4f})")
    
    print("\n=== PUNTOS CLASE 1 ===")
    for i, point in enumerate(ind.points_class1):
        print(f"Punto {i+1}: ({point[0]:.4f}, {point[1]:.4f})")

    # Evaluar fitness
    fitness = ind.compute_fitness(model)
    print(f"\nFitness: {fitness:.4f}")
    print(f"Distancia promedio entre pares: {ind.getDistances():.4f}")
    print(f"Dispersión: {ind.getDispersion():.4f}")

    # Verificar que todos los pares son válidos
    ind.getClasses(model)
    all_valid = True
    for i in range(ind.num_pairs):
        if ind.classes[i] == ind.classes[i + ind.num_pairs]:
            all_valid = False
            print(f"ERROR: Par {i} inválido")
    
    if all_valid:
        print("\nTodos los pares son válidos (clases diferentes)")
    
    #Pa pintar
    #ind.plot_pairs(model, save_path="pares_iniciales.png")



