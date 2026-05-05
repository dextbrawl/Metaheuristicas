import numpy as np
import random
from typing import List, Tuple, Optional
from individual import Individual  

class Population:
    
    def __init__(self, size: int, numPoints: int = 20, limits: Tuple[float, float] = (-20.0, 20.0), model = None):
    
        self.size = size
        self.numPoints = numPoints
        self.limits = limits
        self.model = model
        self.individuals = []
        self.bestFitnessHistory = []
        self.avgFitnessHistory = []
    
    def initializeRandom(self):
        
        self.individuals = []
        
       
        for i in range(self.size):
            ind = Individual(
                numPoints=self.numPoints,
                limits=self.limits,
                model=self.model
            )
            self.individuals.append(ind)
            
            
        
    def evaluateAll(self):
        """
        Evalúa el fitness de todos los individuos de la población
        """
        for ind in self.individuals:
            ind.computeFitness(self.model)
    
    def getBest(self) -> Optional[Individual]:
        if not self.individuals:
            return None
        
        self.evaluateAll()
        best = min(self.individuals, key=lambda ind: ind.fitness)
        return best
    
    
    def tournamentSelection(self, tournamentSize: int = 3) -> Individual:
        tournamentIndices = random.sample(range(len(self.individuals)), tournamentSize)
        tournament = [self.individuals[i] for i in tournamentIndices]
        
        winner = min(tournament, key=lambda ind: ind.fitness)
        return winner
    
    def getStatistics(self) -> dict:
        self.evaluateAll()
        fitnesses = [ind.fitness for ind in self.individuals]
        
        stats = {
            'min': np.min(fitnesses),
            'max': np.max(fitnesses),
            'avg': np.mean(fitnesses),
            'std': np.std(fitnesses)
        }
        return stats
    
    def showStatistics(self):
        """
        Muestra las estadísticas de la población por pantalla
        """
        stats = self.getStatistics()
        print("POBLACION:")
        print(f"  Mejor fitness (min):    {stats['min']:.4f}")
        print(f"  Peor fitness (max):     {stats['max']:.4f}")
        print(f"  Fitness promedio:          {stats['avg']:.4f}")
        print(f"  Desviación estandar:       {stats['std']:.4f}")
    
    def showBestIndividual(self):
        """
        Muestra información detallada del mejor individuo
        """
        best = self.getBest()
        if best:
            print("TOP PANCHITO")
            best.showInfo()


if __name__ == "__main__":
    import prueba as modelo
    print("PRUEBA DE LA CLASE POPULATION")
    model = modelo.BlackBoxModel("blackbox_modelB.pkl")
    
    population = Population(
        size=20,              # 20 individuos
        numPoints=20,         # cada uno con 20 puntos (10 pares)
        limits=(-1.0, 1.0),   # rango [-1, 1] (ajustado a tu modelo)
        model=model
    )
    
    population.initializeRandom()
    population.showStatistics()
    population.showBestIndividual()
    
    print("\nSELECCION DE PADRES TORNEO")
    winner = population.tournamentSelection(tournamentSize=5)
    print(f"Ganador del torneo con fitness: {winner.fitness:.4f}")