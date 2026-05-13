import numpy as np
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import prueba as modelo

def rectaPares(p1, p2):
            
            x1,y1 = p1
            x2,y2 = p2
            if x2 != x1:
                m = (y2-y1)/(x2-x1)  
            else: 
                float('inf') 
            return lambda x: m * (x - x1) + y1

class Individual:

    
    def __init__(self, numPoints: int = 20, limits: Tuple[float, float] = (-2.0, 2.0), model: modelo.BlackBoxModel = None):
        self.numPoints = numPoints
        self.limits = limits
        self.model = model
        
        self.points = np.random.uniform(limits[0], limits[1], (numPoints, 2))
        self.classes = None
        self.fitness = None
        self.pairs = []
        self.components = {}
        
        if model is not None:
            self.getClasses(model)
            self.smartPairing()
    def getClasses(self, model: modelo.BlackBoxModel) -> np.ndarray:
        self.classes = np.array([model.predict(point) for point in self.points])
        return self.classes
    

    def minpairdistance(self,):
        
        for i, j in self.pairs:
            point_a = self.points[i]
            class_a = self.model.predict(point_a)


            point_b = self.points[j]
            class_b = self.model.predict(point_b)


            recta = rectaPares(point_a,point_b) #y = mx + n

            distance = np.linalg.norm(point_a - point_b)

            step = 0.0002

            #Primero ajustamos x

            point_a = np.array([point_a[0] + step,recta(point_a[0] + step)])
            newDistance = np.linalg.norm(point_a - point_b)
            if(newDistance > distance):
                step = step * (-1)

            while(class_a != class_b):
                point_a = [point_a[0] + step,recta(point_a[0] + step)]
                class_a = self.model.predict(point_a)
            
            point_a = np.array([point_a[0] - step,recta(point_a[0] - step)])


            step = step * - 1
            point_b = np.array([point_b[0] + step,recta(point_b[0] + step)])
            newDistance = np.linalg.norm(point_a - point_b)
            
            class_a = self.model.predict(point_a)
            while(class_a != class_b):
                point_b = [point_b[0] + step,recta(point_b[0] + step)]
                class_b = self.model.predict(point_b)
            
            point_b = np.array([point_b[0] - step,recta(point_b[0] - step)])


            self.points[i] = point_a
            self.points[j] = point_b

    

    def randomPairing(self):
        indices = list(range(self.numPoints))
        random.shuffle(indices)
        
        self.pairs = []
        for i in range(0, len(indices), 2):
            if i + 1 < len(indices):
                self.pairs.append((indices[i], indices[i + 1]))
        
        return self.pairs

    def smartPairing(self):
        if self.classes is None:
            self.getClasses(self.model)
        
        indices_class0 = [i for i, c in enumerate(self.classes) if c == 0]
        indices_class1 = [i for i, c in enumerate(self.classes) if c == 1]
        
        random.shuffle(indices_class0)
        random.shuffle(indices_class1)
        
        self.pairs = []
        
        min_len = min(len(indices_class0), len(indices_class1))
        
        for k in range(min_len):
            self.pairs.append((indices_class0[k], indices_class1[k]))
        
        remaining = indices_class0[min_len:] + indices_class1[min_len:]
        
        for i in range(0, len(remaining) - 1, 2):
            self.pairs.append((remaining[i], remaining[i + 1]))
        
        return self.pairs
    
    def varietyPenalty(self) -> float:
        """
        La penalizacion por variedad es para castigar que no haya puntos de ambos tipos
        Contamos los puntos de la clase 0 y le restamos el 50% del numero de puntos.
        Asi, si hay 10 puntos, y 3 son del tipo 0 seria: 3 - 5 = 2/5
        El abs es porque si se pasa tambien falta diversidad.
        """
        countClass0 = np.sum(self.classes == 0)
        
        idealCount = self.numPoints / 2
        penalty = abs(countClass0 - idealCount) / idealCount
        
        return penalty
    
    def averagePairDistance(self) -> float:
        if not self.pairs:
            self.smartPairing()
        
        distances = []
        #Distancia euclidiana en los pares (entre un punto de un par y su simetrico u homologo)
        #Raiz de suma de cuadrados y la media de eso
        for i, j in self.pairs:
            distance = np.linalg.norm(self.points[i] - self.points[j])
            distances.append(distance)
        
        return np.mean(distances)
    
    def dispersion(self) -> float:
        if not self.pairs:
            self.smartPairing()
        
        if len(self.pairs) < 2:
            return 0.0
        
        minDistance = float('inf')
        
        for p1 in range(len(self.pairs)):
            for p2 in range(p1 + 1, len(self.pairs)):
                i1, j1 = self.pairs[p1]
                i2, j2 = self.pairs[p2]
                
                distance1 = np.linalg.norm(self.points[i1] - self.points[i2])
                distance2 = np.linalg.norm(self.points[i1] - self.points[j2])
                distance3 = np.linalg.norm(self.points[j1] - self.points[i2])
                distance4 = np.linalg.norm(self.points[j1] - self.points[j2])
                
                closestDistance = min(distance1, distance2, distance3, distance4)
                
                if closestDistance < minDistance:
                    minDistance = closestDistance
        
        return minDistance if minDistance != float('inf') else 0.0
    
    def sameClassPenalty(self) -> float:
        if not self.pairs:
            self.smartPairing()
        
        #Si en un par son de la misma clase se suma 1 al contador, y se penaliza al individuo contador/numero de pares
        sameClassCount = 0
        for i, j in self.pairs:
            if self.classes[i] == self.classes[j]:
                sameClassCount += 1
        
        penalty = sameClassCount / len(self.pairs)
        
        return penalty
    
    def computeFitness(self, model: modelo.BlackBoxModel, 
                       weightDistance: float = 1.0, #Distancia media de los puntos de los pares
                       weightDispersion: float = 0.5, #Distancia entre pares
                       weightVariety: float = 1.0, #Variedad
                       weightSameClass: float = 100.0) -> float:
        """
        Fitness = distanciaPromedio 
                -dispersion 
                + penalizacionVariedad 
                + penalizacionMismaClase
        """
        self.getClasses(model)
        
        avgDistance = self.averagePairDistance()
        dispersionValue = self.dispersion()
        varietyPenalty = self.varietyPenalty()
        sameClassPenalty = self.sameClassPenalty()
        
        self.fitness = (weightDistance * avgDistance 
                       - weightDispersion * dispersionValue 
                       + weightVariety * varietyPenalty 
                       + weightSameClass * sameClassPenalty)
        
        self.components = {
            'avgDistance': avgDistance,
            'dispersion': dispersionValue,
            'varietyPenalty': varietyPenalty,
            'sameClassPenalty': sameClassPenalty
        }
        
        return self.fitness

    def showInfo(self):
        """
        Muestra información detallada del individuo
        """
        if self.classes is None:
            print("Individuo no evaluado. Ejecuta computeFitness() primero.")
            return
        
        countClass0 = np.sum(self.classes == 0)
        countClass1 = np.sum(self.classes == 1)
        
        print("INDIVIDUO:")
        print(f"Puntos totales: {self.numPoints}")
        print(f"Clase 0: {countClass0} puntos")
        print(f"Clase 1: {countClass1} puntos")
        print(f"Numero de pares: {len(self.pairs)}")
        
        # Contar pares por tipo
        sameClass = 0
        diffClass = 0
        for i, j in self.pairs:
            if self.classes[i] == self.classes[j]:
                sameClass += 1
            else:
                diffClass += 1
        
        print(f"Pares de misma clase: {sameClass}")
        print(f"Pares de clases diferentes: {diffClass}")
        
        print("\n--- COMPONENTES DEL FITNESS ---")
        print(f"Distancia promedio entre pares: {self.components.get('avgDistance', 0):.4f}")
        print(f"Dispersión (distancia mínima entre pares): {self.components.get('dispersion', 0):.4f}")
        print(f"Penalización por variedad: {self.components.get('varietyPenalty', 0):.4f}")
        print(f"Penalización por misma clase: {self.components.get('sameClassPenalty', 0):.4f}")
        print(f"\nFITNESS TOTAL: {self.fitness:.4f} (menor es mejor)")    
    
    def getAproximationPoints(self):
        
        aprox_points = []
        
        for i, j in self.pairs:
            
            point_a = self.points[i]
            point_b = self.points[j]
            
            class_point_a = self.model.predict(point_a)
            class_point_b = self.model.predict(point_b)
            
            if(class_point_a != class_point_b):
                aprox_points.append([((point_a[0] + point_b[0])/2), ((point_a[1] + point_b[1])/2)])
            
        return np.array(aprox_points)

def mutate(ind: Individual, mRate):
    """
    ind: Individuo a mutar

    mRate: Ratio de mutación, máximo de varianza de la mutación. mRate = 0 -> no mutación
    """

    newPoints = ind.points.copy()

    for a in newPoints:
        a[0] += mRate * random.uniform(-1,1)

    retVal = Individual()

    retVal.points = newPoints

    return retVal
        
if __name__ == "__main__":
    model = modelo.BlackBoxModel("blackbox_modelB.pkl")
    
    print("CREANDO INDIVIDUO CON PUNTOS ALEATORIOS")

    myIndividual = Individual(numPoints=20, limits=(-2.0, 2.0), model=model)
    fitnessValue = myIndividual.computeFitness(model)
    myIndividual.minpairdistance()
    
    print("\n--- PUNTOS GENERADOS ---")
    for i, point in enumerate(myIndividual.points):
        pointClass = myIndividual.classes[i]
        print(f"Punto {i+1:2d}: ({point[0]:7.4f}, {point[1]:7.4f}) -> Clase {int(pointClass)}")

    print("\n--- EMPAREJAMIENTO ALEATORIO ---")
    for k, (i, j) in enumerate(myIndividual.pairs):
        classI = myIndividual.classes[i]
        classJ = myIndividual.classes[j]
        pairType = "DIFERENTE" if classI != classJ else "MISMA"
        print(f"Par {k+1:2d}: Punto {i+1:2d} (clase {int(classI)}) con Punto {j+1:2d} (clase {int(classJ)}) -> {pairType}")