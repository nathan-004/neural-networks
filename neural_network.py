import math
import json
from copy import deepcopy
import time
import numpy as np

class NeuralNetwork():
    pass

class GeneticAI():
    pass

# -----------------------Activation Functions-------------------------
def relu(z):
    """Rectified Linear Units (ReLU)"""
    return np.maximum(0, z)

def sigmoid(z):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-z))

def linear(z):
    return z

def tanh(x):
    return np.tanh(x)

# ----------------------Cost Functions------------------------------
def erreur_relative(liste1, liste2):
    erreur_absolue = [abs(a - b) for a, b in zip(liste1, liste2)]
    erreurs_relatives = [erreur / a for erreur, a in zip(erreur_absolue, liste1)]
    return sum(erreurs_relatives) / len(erreurs_relatives)

def mse(preds, targets):
    """Mean Squared Error"""
    return sum((p - t)**2 for p, t in zip(preds, targets)) / len(preds)

def bce(y_pred, y_true):
    epsilon = 1e-7
    losses = []
    for yt, yp in zip(y_true, y_pred):
        yp = max(min(yp, 1 - epsilon), epsilon)
        loss = - (yt * math.log(yp) + (1 - yt) * math.log(1 - yp))
        losses.append(loss)
    return sum(losses) / len(losses)

def binary_error(preds, targets):
    loss = []
    for targ, pred in zip(targets, preds):
        loss.append(abs(targ-pred))
    return sum(loss) / len(loss)

# --------------------------- Neural Network Operations ---------------------------------
def copy(neural_network1, neural_network2):
    """
    Copie les poids et biais de neural_network1 vers neural_network2.

    Parameters
    ----------
    neural_network1 : NeuralNetwork
    neural_network2 : NeuralNetwork
    """

    for i in range(len(neural_network1.weights)):
        neural_network2.weights[i] = np.copy(neural_network1.weights[i])
        neural_network2.biases[i] = np.copy(neural_network1.biases[i])


def crossover(nn1: NeuralNetwork, nn2: NeuralNetwork):
    """Croisement entre deux réseaux : Moyenne des poids/biais"""
    child = NeuralNetwork(
        n_layers=len(nn1.weights) - 1,
        hidden_size=nn1.weights[0].shape[1],  # nombre de neurones dans les couches cachées
        n_input=nn1.weights[0].shape[1],
        n_output=nn1.weights[-1].shape[0],
        hidden_activation_function=nn1.hidden_activation_function,
        output_activation_function=nn1.output_activation_function,
    )

    # Crossover sur chaque couche
    for i in range(len(nn1.weights)):
        child.weights[i] = (nn1.weights[i] + nn2.weights[i]) / 2
        child.biases[i] = (nn1.biases[i] + nn2.biases[i]) / 2

    return child


def save_population(population:list, epochs:int, errors:list, t:int, filename="neural_networks.json"):
    """
    Enregistre la population de Neural Networks dans un fichier JSON
    
    Parameters
    ----------
    population:list
        Liste contenant des NeuralNetwork
    epochs:int
        Nombre d'itération sur les données d'entraînement
    errors:list
        Liste des meilleurs erreurs dans l'entraînement
    """
    stock = {
        "Parameters" : {
            "Epochs": epochs,
            "Population_Size": len(population),
            "Training_Time": t,
            "Training_Time_m": t // 60,
            "Errors": errors,
        }
    }

    for idx, nn in enumerate(population):
        stock[str(idx)] = nn.export()

    with open(filename, "w") as f:
        json.dump(stock, f, indent=4)
        
def import_data(filename="neural_networks.json") -> dict:
    """Retourne les données sous forme de dictionnaire"""
    with open(filename, "r") as f:
        dictionnaire = json.load(f)

    return dictionnaire

# ----------------------------------Training Operations-------------------------------------
def curriculum_learning(model:GeneticAI, total_epochs:int, training_data:list, limites:list, filename=""):
    """
    Entraîne le modèle sur une petite quantité de donnée puis augmente la taille

    Parameters
    ----------
    model:GeneticAI
    total_epochs:int
        Nombre total d'itérations
    training_data:list
        Données d'entraînement
    limites:list
        Listes de nombres entre 0 et 1 définissant le pourcentage de données utilisées dans training_data
    """

    for idx, limite in enumerate(limites):
        limite_idx = int(len(training_data) * limite)
        data = training_data[:limite_idx]
        epoch = total_epochs // len(limites)

        errors = model.train(data, epoch, mutation_base=5, croisement=False, debug=False, filename="")
    
    return errors

class NeuralNetwork:
    """Réseau neuronal qui utilise des matrices de poids et de biais au lieu d'objets Node"""
    
    min_weight = -1
    max_weight = 1
    min_bias = -1
    max_bias = 1
    activation_functions = {
        "relu": relu,
        "sigmoid": sigmoid,
        "linear": linear,
        "tanh": tanh,
    }

    def __init__(self, n_layers, hidden_size, n_input, n_output, hidden_activation_function="relu", output_activation_function="linear"):
        if hidden_activation_function not in self.activation_functions or output_activation_function not in self.activation_functions:
            raise ValueError("La fonction d'activation spécifiée n'existe pas")
        
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

        # Exemple : [8, 10, 10, 1] → input, 2 couches cachées, output
        layers = [n_input] + [hidden_size] * n_layers + [n_output]

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            W = np.random.uniform(self.min_weight, self.max_weight, (layers[i + 1], layers[i]))
            b = np.random.uniform(self.min_bias, self.max_bias, (layers[i + 1], 1))
            self.weights.append(W)
            self.biases.append(b)

    def prediction(self, inputs):
        """
        Calcule la sortie du réseau avec les entrées.
        inputs: liste de valeurs d'entrée
        """
        a = np.array(inputs).reshape(-1, 1)  # vecteur colonne

        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(W, a) + b
            if idx < len(self.weights) - 1:
                activation = self.activation_functions[self.hidden_activation_function]
            else:
                activation = self.activation_functions[self.output_activation_function]
            a = activation(z)

        return a.flatten().tolist()
    
    def mutate(self, error, mutation_base=None):
        """
        Fait muter les poids et les biais aléatoirement selon une intensité liée à l'erreur.

        Parameters
        ----------
        error : float
            Erreur absolue
        mutation_base : float, optional
            Valeur de base pour la mutation (amplitude max). Si None, on utilise self.variation_base.
        """
        if mutation_base is None:
            base_mutation = self.variation_base
        else:
            base_mutation = mutation_base

        cur_variation = base_mutation * abs(error)

        # Mutation aléatoire sur tous les poids et biais
        for i in range(len(self.weights)):
            variation_W = np.random.uniform(-cur_variation, cur_variation, self.weights[i].shape) # Variation
            variation_b = np.random.uniform(-cur_variation, cur_variation, self.biases[i].shape)

            self.weights[i] += variation_W
            self.biases[i] += variation_b

    def export(self):
        """
        Retourne les poids et biais sous forme de liste pour chaque couche (sauf input)

        Returns
        -------
        list
            Liste de tuples (poids, biais) pour chaque couche. Chaque poids est une matrice NumPy, chaque biais est un vecteur.
        """
        export = []
        for W, b in zip(self.weights, self.biases):
            export.append([W.tolist(), b.tolist()])
        return export

    def import_nn(self, layers):
        """
        Importe les poids et biais d'un réseau de neurones vectorisé.

        Parameters
        ----------
        layers : list
            Liste contenant les poids et biais sous forme de listes (générés par la méthode export()).
        """
        self.weights = []
        self.biases = []
        
        for W_list, b_list in layers:
            W = np.array(W_list)
            b = np.array(b_list)
            self.weights.append(W)
            self.biases.append(b)

        
class GeneticAi:

    error_functions = {
        "mse": mse,
        "erreur_relative": erreur_relative,
        "bce": bce,
        "erreur_binaire": binary_error,
    }
    
    start_epoch = 0 # Epoch à commencer avec
    start_errors = [] # Liste des erreurs à poursuivre
    start_time = 0

    def __init__(self, population_size, n_layers,hidden_size, n_input, n_output, erreur_calcul="mse", hidden_activation_function:str="relu", output_activation_function="linear"):
        """
        Parameters
        ----------
        population_size:int
            Nombre de Réseau de neurones
        n_layers:int
            Nombre de couches
        hidden_size:int
            Nombre de noeuds pour chaques couches
        n_input:int
            Taille de l'input
        n_output:int
            Taille de l'output
        """
        self.population_size = population_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_input = n_input
        self.n_output = n_output
        self.error = self.error_functions[erreur_calcul]
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

        self.population = [NeuralNetwork(n_layers, hidden_size, n_input, n_output, hidden_activation_function, output_activation_function) for _ in range(population_size)] # Liste des réseaux de Neurones

    def train(self, training_data:dict, epochs:int, mutation_base:float=0.1, nn_stockage=None, croisement=False, debug=False, filename=None):
        """
        Calcule la précision de chaque Réseau et garde les meilleurs pour les modifier epochs fois
        """
        if filename is not None:
            if filename != "":
                self.import_population(filename)
            else:
                self.import_population()
        
        errors = self.start_errors # Passage par référence
        start = time.time()
        try:
            for i in range(epochs):
                i += self.start_epoch
                if debug:
                    t0 = time.time()
                # Faire une liste triés du meilleur au pire et ne garder que les 1/10
                if debug:
                    t1 = time.time()
                precisions = {nn: self.get_precision(nn, training_data) for nn in self.population}
                if debug:
                    t2 = time.time()
                    print(f"[DEBUG] Calcul des précisions : {t2-t1:.4f} s")
                precisions_sorted = sorted(precisions.items(), key=lambda x: x[1])
                n_best = max(1, self.population_size // 3) # 1/3
                best_networks = [nn for nn, _ in precisions_sorted[:n_best]]
                best = best_networks[0]
                err  = precisions[best]
                errors.append(err)
                print(f"Epoch {i+1}/{epochs+self.start_epoch} — meilleure erreur = {err:.4f}")

                if debug:
                    t3 = time.time()
                    print(f"[DEBUG] Tri et sélection des meilleurs : {t3-t2:.4f} s")

                # Recréer la population avec les 1/10 qui sont les mêmes et les autres sont des dérivés des 1/10 meilleurs
                best_index = 0 # Index de best_networks

                for idx, nn in enumerate(self.population):
                    if idx < n_best:
                        self.population[idx] = deepcopy(best_networks[idx])
                        continue

                    best_net = best_networks[best_index]
                    error = precisions[best_net]
                    if not croisement:
                        copy(best_net, nn)
                    else:
                        if best_index == len(best_networks) - 1:
                            copy(crossover(best_net, best_networks[0]), nn)
                        else:
                            copy(crossover(best_net, best_networks[best_index+1]), nn)
                    nn.mutate(error, mutation_base)
                
                    best_index += 1

                    if best_index == n_best:
                        best_index = 0

                if debug:
                    t4 = time.time()
                    print(f"[DEBUG] Recréation et mutation de la population : {t4-t3:.4f} s")
                    print(f"[DEBUG] Temps total pour l'epoch {i+1}: {t4-t0:.4f} s\n")
        except KeyboardInterrupt:
            pass

        duration = self.start_time + time.time() - start
        if nn_stockage is None:
            save_population(self.population, i, errors, duration)
        else:
            save_population(self.population, i, errors, duration, nn_stockage)

        return errors

    def get_precision(self, neural_network:NeuralNetwork, training_data:dict):
        """
        Parameters
        ----------
        neural_network:NeuralNetwork
        training_data:dict
            Dictionnaire sous la forme {input:tuple, outputs:tuple}

        Returns
        -------
        Moyenne des erreurs relatives des différents inputs et outputs
        """

        results = 0

        for input_, output in training_data.items():
            pred = neural_network.prediction(input_)
            results += self.error(pred, output) # Calculer erreur relative

        return results / len(training_data)
    
    def import_population(self, filename="neural_networks.json"):
        """
        Crée une population basée sur le contenu du fichier
        
        Parameters
        ----------
        filename:str
        """
        
        data = import_data(filename)
        header = data.pop("Parameters")
        
        for current_nn, values in zip(self.population, data) :
            current_nn.import_nn(data[values])

        self.start_epoch = header["Epochs"]
        self.start_errors = header["Errors"]
        self.start_time = header["Training_Time"]

if __name__ == "__main__":
    pass
