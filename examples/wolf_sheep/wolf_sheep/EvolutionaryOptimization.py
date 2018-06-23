from inspyred import ec
from random import Random
from time import time
from fitness import *
from numpy import genfromtxt
import matplotlib.pyplot as plt

### Optimisation Evolutionnaire avec inspyred
def indvGenerator(random,args): #Génére un individu aléatoire
    lowBound=args['_ec'].bounder.lower_bound
    upperBound=args['_ec'].bounder.upper_bound
    return np.random.uniform(low=lowBound, high=upperBound, size=args['nDim'])


def plus_replacement(random, population, parents, offspring, args):
    pool = list(offspring)
    pool.extend(population)
    pool.sort(reverse=True)
    survivors = pool[:len(population)]
    return survivors


#Fixe le générateur de nombre aléatoire
prng = Random()
prng.seed(time())


# Création et paramétrisation de l'algorithme évolutionnaire
ea = ec.EvolutionaryComputation(prng)
ea.selector = ec.selectors.tournament_selection # Choix de la fonction de selection
ea.variator = [ec.variators.blend_crossover, #Choix de la fonction de crossover
               ec.variators.gaussian_mutation] # Choix de la fonction de mutation
ea.replacer = plus_replacement #Choix de la fonction de remplacement générationnel
ea.terminator = ec.terminators.generation_termination # Choix de la fonction qui arrête l'algorithme
ea.observer = [ec.observers.stats_observer,ec.observers.file_observer] #Permet de récolter des statistiques à chaque génération
popsize=20#Taille de la population
nVar=1 # Nombre de dimension du problème
maxIt = 100
minBound=np.repeat(0,nVar) # Bornes min
maxBound=np.repeat(1,nVar) # Bornes max
statsFile=open('stats_pop.csv','w')
indvsFile=open('IndvsForAllGens.csv','w')

#Lance l'optimisation et retourne la dernière génération
final_pop = ea.evolve(generator=indvGenerator, #Générateur d'individu
                      evaluator=ec.evaluators.evaluator(fitness), #Evaluateur de population
                      pop_size=popsize, #Taille de la population
                      bounder=ec.Bounder(minBound,maxBound), #Définition des bornes
                      maximize=True, #On cherche ici à minimiser la fonction
                      tournament_size=2, #Taille de tournoi
                      num_selected=popsize, #Nombre de parents selectionnés pour la production d'enfants. Est égal au nombre d'enfants générés
                      max_generations=maxIt, #Nombre de générations max
                      mutation_rate=0.1, #Proportion de mutation
                      blx_alpha=0.2, #Paramètre alpha pour le crossover
                      gaussian_stdev=3, #Paramètre de déviation standard pour la mutation gaussienne
                      nDim=nVar, #Nombre de dimensions du problème
                      statistics_file=statsFile, #Fichier où l'on enregistre les statistiques des différentes générations
                      individuals_file=indvsFile) #Fichier où l'on enregistre les individus de chaque génération


statspop_data = genfromtxt('stats_pop.csv', delimiter=',')
plt.plot(np.arange(popsize,(maxIt+2)*popsize,popsize),statspop_data[:,3],'r*-')
plt.xlabel("Nombre d'évaluations")
plt.ylabel("Fitness")
plt.title("Meilleure fitness vue au cours des évaluations")
plt.legend()
plt.show()