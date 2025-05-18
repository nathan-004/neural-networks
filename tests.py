from neural_network import *

import matplotlib.pyplot as plt

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
        (6,): (12,),
        (7,): (14,),
        (10,): (20,),
        (15,): (30,)
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
        (13,): (169,),
        (14,): (196,),
    },
    "CtoF" : { # celsius to fahrenheit
        (0,): (32,),
        (10,): (50,),
        (20,): (68,),
        (30,): (86,),
        (40,): (104,),
        (100,): (212,),
        (-40,): (-40,),
        (-10,): (14,),
        (25,): (77,),
        (37,): (98.6,),
    },
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
    
    ai.train(training_data, epochs=10000)

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
        erreur_calcul="erreur_relative",
    )
    
    ai.train(training_data, epochs=3000)

    for i in range(15):
        print(i, ai.population[0].prediction([i])[0], i**2, sep=" : ")

    graph_results(ai.population[0], lambda x : x**2, -15, 15)

if __name__ == "__main__":
   celsius_to_fahrenheit()
