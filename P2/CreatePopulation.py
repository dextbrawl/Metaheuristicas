import random
import numpy as np
import individuals as ind

#Metodo para crear un individuo aleatorio
def CreateRandomIndividual(population_size):
    individual = ind.Individual(
            n_estimators=random.randint(10, 300),
            max_depth=random.randint(2, 30),
            min_samples_split=random.randint(2, 20),
            min_samples_leaf=random.randint(1, 20),
            max_features=round(random.uniform(0.1, 1.0), 2),
            bootstrap=random.choice([0, 1]),
            criterion=random.choice([0, 1]),
            class_weight=random.choice([0, 1]),
            max_leaf_nodes=random.randint(10, 200),
            min_impurity_decrease=round(random.uniform(0, 0.1), 3),
            random_state=random.randint(1, 1000),
            position=population_size+1
        )
    return individual

#Dado un individuo, normalizamos en [0,1] para poder calcular distancias
def NormaliseIndividual(ind):
    return np.array([
        ind.n_estimators / 300.0,      
        ind.max_depth / 30.0,
        ind.min_samples_split / 20.0,
        ind.min_samples_leaf / 20.0,
        ind.max_features,              
        ind.bootstrap,                  
        ind.criterion,                  
        ind.class_weight,               
        ind.max_leaf_nodes / 200.0,
        ind.min_impurity_decrease / 0.1,
        ind.random_state / 1000.0
    ])


#Dado un nuevo individuo, se halla su distancia minima al resto de la poblacion
def PrimeDistance(NewIndividual, AlreadyExistingIndividuals):
    if not AlreadyExistingIndividuals:
        return float('inf')
    
    NormalisedIndividual = NormaliseIndividual(NewIndividual)
    MinDistance = float('inf')
    
    for Existing in AlreadyExistingIndividuals:
        existing_vec = NormaliseIndividual(Existing)
        EuclideanDistance = np.sqrt(np.sum((NormalisedIndividual - existing_vec) ** 2)) #Distancia euclidiana
        
        if EuclideanDistance < MinDistance:
            MinDistance = EuclideanDistance
    
    return MinDistance

#Crear una poblacion random de tamaño population
def CreateRandomPopulation(population):
    poblacion = []
    for _ in range(population):
        individuo = CreateRandomIndividual()
        poblacion.append(individuo)
    return poblacion


#Poblacion inicial secuencial
#Distancia minima 0.15 y maximo de intentos 100
def CreateSequentialPopulation(population, Min, MaxTry): 
    poblacion = []
    
    for i in range(population):
        attempts = 0
        nuevo_individuo = None
        
        while attempts < MaxTry:
            NewIndividual = CreateRandomIndividual()
            
            distancia_min = PrimeDistance(NewIndividual, poblacion)
            
            if i == 0 or distancia_min >= Min:
                poblacion.append(NewIndividual)
                break
            
            attempts += 1
        
        if attempts == MaxTry:
            print(f"Individuo {i+1} no cumple distancia minima de {Min}")
            print(f"Distancia mínima alcanzada: {distancia_min:.4f}")
            poblacion.append(NewIndividual)
    
    return poblacion



