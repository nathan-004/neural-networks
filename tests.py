from neural_network import *

import matplotlib.pyplot as plt
from math import sin
from csv import DictReader

def importer_table(fichier):
    with open(fichier, encoding="UTF-8") as f:
        u = []
        for dict in DictReader(f, delimiter=","):
            for el in dict:
                dict[el] = float(dict[el])
            u.append(dict)
    return u

def graph_results(model:NeuralNetwork, true_func, min:int, max:int):
    """
    Affiche un graphique comparant les prédictions d'un réseau de neurones à une fonction réelle.

    Parameters
    ----------
    model : NeuralNetwork
        Le modèle de réseau de neurones à tester.
    true_func : callable
        La fonction réelle à comparer (par exemple, lambda x: x**2).
    min : int
        La valeur minimale de x à tester (incluse).
    max : int
        La valeur maximale de x à tester (exclue).

    Affiche
    -------
    Un graphique matplotlib avec :
        - La courbe de la vraie fonction (en bleu)
        - La courbe des prédictions du réseau de neurones (en orange pointillé)
    """

    xs = [i+min for i in range(max-min)]
    reals = [true_func(min+x) for x in range(max-min)]
    preds = [model.prediction([x+min])[0] for x in range(max-min)]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, reals, label="Vraie fonction", color="blue", linewidth=2)
    plt.plot(xs, preds, label="Prédiction NN", color="orange", linestyle="--")
    plt.title("Comparaison des prédictions vs vraie fonction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

training_datas = {
    "Mult2" : {
        (0,): (0,),
        (1,): (2,),
        (2,): (4,),
        (3,): (6,),
        (4,): (8,),
        (5,): (10,),
        (6,): (12,),
        (7,): (14,),
        (8,): (16,),
        (9,): (18,),
        (10,): (20,),
        (12,): (24,),
        (15,): (30,),
        (20,): (40,),
        (-1,): (-2,),
        (-5,): (-10,),
        (-10,): (-20,),
    },
    "Pow2" : {
        (0,): (0,),
        (1,): (1,),
        (2,): (4,),
        (3,): (9,),
        (4,): (16,),
        (5,): (25,),
        (6,): (36,),
        (7,): (49,),
        (8,): (64,),
        (9,): (81,),
        (10,): (100,),
        (11,): (121,),
        (12,): (144,),
        (13,): (169,),
        (-1,): (1,),
        (-2,): (4,),
        (-3,): (9,),
        (-4,): (16,),
    },
    "CtoF" : { # celsius to fahrenheit
        (0,): (32,),
        (10,): (50,),
        (20,): (68,),
        (30,): (86,),
        (40,): (104,),
        (50,): (122,),
        (60,): (140,),
        (70,): (158,),
        (80,): (176,),
        (90,): (194,),
        (100,): (212,),
        (-40,): (-40,),
        (-30,): (-22,),
        (-20,): (-4,),
        (-10,): (14,),
        (25,): (77,),
        (37,): (98.6,),
    },
    "Sin" : {
        (0,): (sin(0),),
        (1,): (sin(1),),
        (2,): (sin(2),),
        (3,): (sin(3),),
        (4,): (sin(4),),
        (5,): (sin(5),),
        (6,): (sin(6),),
        (7,): (sin(7),),
        (8,): (sin(8),),
        (9,): (sin(9),),
        (10,): (sin(10),),
        (-1,): (sin(-1),),
        (-2,): (sin(-2),),
        (-3,): (sin(-3),),
        (-4,): (sin(-4),),
        (-5,): (sin(-5),),
        (-6,): (sin(-6),),
        (-7,): (sin(-7),),
        (-8,): (sin(-8),),
        (-10,): (sin(-10),),
    }
}

def celsius_to_fahrenheit():
    training_data = training_datas["CtoF"]

    ai = GeneticAi(population_size=200,
               n_layers=0,
               hidden_size=5,
               n_input=1,
               n_output=1)
    
    ai.train(training_data, epochs=2000)
    for c in [-40, -10, 0, 10, 20, 25, 30, 37, 40, 100]:
        print(f"{c} : {ai.population[0].prediction([c])[0]} : {32 + c * 9/5:.2f}")

    print(ai.population[0].output_layer[0].weights, ai.population[0].output_layer[0].bias)
    graph_results(ai.population[0], lambda x : 32+x*(9/5), -100, 100)

def multiplication_2():
    training_data = training_datas["Mult2"]

    ai = GeneticAi(population_size=100,
               n_layers=0,
               hidden_size=5,
               n_input=1,
               n_output=1)
    
    ai.train(training_data, epochs=1000)

    for i in [-40, -10, 0, 10, 20, 25, 30, 37, 40, 100]:
        print(i, ai.population[0].prediction([i])[0], i*2, sep=" : ")
    
    print(ai.population[0].output_layer[0].weights, ai.population[0].output_layer[0].bias)

    graph_results(ai.population[0], lambda x : x*2, -40, 100)

def power_2():
    training_data = training_datas["Pow2"]

    ai = GeneticAi(
        population_size=100,
        n_layers=3,           # 1 couche cachée
        hidden_size=10,
        n_input=1,
        n_output=1,
        erreur_calcul="mse",
    )
    
    ai.train(training_data, epochs=1000)

    for i in range(15):
        print(i, ai.population[0].prediction([i])[0], i**2, sep=" : ")

    graph_results(ai.population[0], lambda x : x**2, -15, 15)

def sin_x():
    training_data = training_datas["Sin"]

    ai = GeneticAi(
        population_size=100,
        n_layers=3,
        hidden_size=10,
        n_input=1,
        n_output=1,
        erreur_calcul="mse",
    )

    errors = ai.train(training_data, epochs=750)

    for i in range(15):
        print(i, ai.population[0].prediction([i])[0], sin(i), sep=" : ")

    graph_results(ai.population[0], lambda x : sin(x), -20, 20)
    
    xs = [i for i in range(len(errors))]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, errors, label="Erreur", color="blue", linewidth=2)
    plt.title("Comparaison des prédictions vs vraie fonction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def classification_diabete():
    data = importer_table("data/diabetes.csv")
    limite = int(len(data) * 0.2)
    training_data = {
        tuple(list(line.values())[:-1]): (float(list(line.values())[-1]),)
        for line in data[:limite]
    }

    test_data = {
        tuple(float(x) for x in list(row.values())[:-1]):
        (float(list(row.values())[-1]),)
        for row in data[limite:]
    }

    ai = GeneticAi(
        population_size=50,
        n_layers=3,
        hidden_size=6,
        n_input=8,
        n_output=1,
        erreur_calcul="erreur_binaire",
        hidden_activation_function="tanh",
        output_activation_function="sigmoid",
    )

    errors = ai.train(training_data, 150, mutation_base=5)
    
    xs = [i for i in range(len(errors))]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, errors, label="Erreur", color="blue", linewidth=2)
    plt.title("Comparaison des prédictions vs vraie fonction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Évaluation sur les données de test
    score = ai.get_precision(ai.population[0], test_data)
    print(f"Erreur moyenne sur test_data : {score:.4f}")

if __name__ == "__main__":
   classification_diabete()
