import numpy as np
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import prueba as modelo

#Un individuo es un conjunto de puntos (de pares de puntos)
class Individual:
    
    def __init__(self, numpairs: int = 15, bounds: Tuple[float, float] = (-50.0, 50.0), model: modelo.BlackBoxModel = None):
        self.numpairs = numpairs
        self.bounds = bounds
        self.model = model
        
        if model is not None:
            self.pairing()
        else:
            self.points0 = np.random.uniform(bounds[0], bounds[1], (numpairs, 2))
            self.points1 = np.random.uniform(bounds[0], bounds[1], (numpairs, 2))
            self.points = np.vstack([self.points0, self.points1])
            self.classes = None
            self.fitness = None
    
    def pairing(self):
       
        self.points0 = []
        self.points1 = []
        
        maxattempts = self.numpairs * 200  # Aumentamos intentos
        attempts = 0
        
        while len(self.points0) < self.numpairs and attempts < maxattempts:
            a = np.random.uniform(self.bounds[0], self.bounds[1], 2)
            b = np.random.uniform(self.bounds[0], self.bounds[1], 2)
            
            class_a = self.model.predict(a)
            class_b = self.model.predict(b)
            
            if class_a != class_b:
                if class_a == 0:
                    self.points0.append(a)
                    self.points1.append(b)
                else:
                    self.points0.append(b)
                    self.points1.append(a)
            
            attempts += 1
        
        # Si no se generaron suficientes pares, completar con puntos aleatorios
        # y luego ajustar sus clases
        while len(self.points0) < self.numpairs:
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
                        self.points0.append(point)
                        self.points1.append(other_point)
                    else:
                        self.points0.append(other_point)
                        self.points1.append(point)
                    break
        
        self.points0 = np.array(self.points0[:self.numpairs])
        self.points1 = np.array(self.points1[:self.numpairs])
        self.points = np.vstack([self.points0, self.points1])
        
        print(f"Generados {len(self.points0)}/{self.numpairs} pares validos")
    
    def getClasses(self, model: modelo.BlackBoxModel) -> np.ndarray:
        self.classes = np.array([model.predict(point) for point in self.points])
        return self.classes
    
    def getDistances(self) -> float:
        if self.classes is None:
            if self.model:
                self.getClasses(self.model)
            else:
                raise ValueError("primero clases bro")
        
        distances = []
        for i in range(self.numpairs):
            dist = np.linalg.norm(self.points0[i] - self.points1[i]) #distancia euclides en R2
            distances.append(dist)
        
        return np.mean(distances)
    
    #Dispersion alta god dispersion baja antigod
    def getDispersion(self) -> float:
        if len(self.points) < 2:
            return 0.0
        
        minDistance = float('inf')
        
        #linalg es la raiz de la suma de cuadrados 
        for i in range(self.numpairs):
            for j in range(i + 1, self.numpairs):
                d00 = np.linalg.norm(self.points0[i] - self.points0[j])
                d11 = np.linalg.norm(self.points1[i] - self.points1[j])
                d01 = np.linalg.norm(self.points0[i] - self.points1[j])
                d10 = np.linalg.norm(self.points1[i] - self.points0[j])
                
                for d in [d00, d11, d01, d10]:
                    if d < minDistance:
                        minDistance = d
        
        return minDistance if minDistance != float('inf') else 0.0
    
    def computeFitness(self, model: modelo.BlackBoxModel) -> float:
       
        #Distancia pares - dispersion
        #Lo suyo es que la distancia sea minima y la dispersion maxima

        self.getClasses(model)
        distance = self.getDistances()
        dispersion = self.getDispersion()
        self.fitness = distance - dispersion
        return self.fitness
    






#ESTE MAIN CREA 1 INDIVIDUO Y LO SACA POR PANTALLA
if __name__ == "__main__":

    model = modelo.BlackBoxModel("blackbox_modelB.pkl")
    
    print("Creando individuo...")
    ind = Individual(numpairs=10, model=model)
    
    print(f"\nIndividuo creado con {ind.numpairs} pares")
    print(f"Puntos clase0: {ind.points0.shape}")
    print(f"Puntos clase1: {ind.points1.shape}")
    
    # Mostrar puntos
    print("\n--- PUNTOS CLASE 0 ---")
    for i, point in enumerate(ind.points0):
        print(f"Punto {i+1}: ({point[0]:.4f}, {point[1]:.4f})")
    
    print("\n--- PUNTOS CLASE 1 ---")
    for i, point in enumerate(ind.points1):
        print(f"Punto {i+1}: ({point[0]:.4f}, {point[1]:.4f})")

    # Evaluar fitness
    fitness = ind.computeFitness(model)
    print(f"\nFitness: {fitness:.4f}")
    print(f"Distancia promedio entre pares: {ind.getDistances():.4f}")
    print(f"Dispersion: {ind.getDispersion():.4f}")

    # Verificar que todos los pares son validos
    ind.getClasses(model)
    valid = True
    for i in range(ind.numpairs):
        if ind.classes[i] == ind.classes[i + ind.numpairs]:
            valid = False
            print(f"ERROR: Par {i} invalido")
    
    if valid:
        print("\nTodos los pares son validos (clases diferentes)")
    
    #Pa pintar
    #ind.plot_pairs(model, save_path="pares_iniciales.png")



