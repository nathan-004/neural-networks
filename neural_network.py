import random
import math
from math import exp
import json

# -----------------------Activation Functions-------------------------
def relu(z):
    """Rectified Linear Units (ReLU)"""
    return max(0, z)

def sigmoid(z):
    """Sigmoid activation"""
    return 1 / (1 + math.exp(-z))

def linear(z):
    return z

def tanh(x):
    return math.tanh(x)

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
        loss.append(int(abs(targ-pred) < 0.5))
    return sum(loss) / len(loss)

# --------------------------- Neural Network Operations ---------------------------------
def copy(neural_network1, neural_network2=None):
    """
    Change the values of the neural_network2 with the values of the neural_network1
    
    If neural_network2 is not specified : creates a new neural_network
    """
    if neural_network2 is None:
        raise AssertionError("A implementer")

    for idx, layer in enumerate(neural_network1.layers):
        for idx_node, node in enumerate(layer):
            neural_network2.layers[idx][idx_node].weights = node.weights.copy()
            neural_network2.layers[idx][idx_node].bias = node.bias

    for idx, node in enumerate(neural_network1.output_layer):
        neural_network2.output_layer[idx].weights = node.weights.copy()
        neural_network2.output_layer[idx].bias = node.bias
        
def save_population(population:list, epochs:int, filename="neural_networks.json"):
    """
    Enregistre la population de Neural Networks dans un fichier JSON
    
    Parameters
    ----------
    population:list
        Liste contenant des NeuralNetwork
    epochs:int
        Nombre d'itération sur les données d'entraînement
    """
    

class Node:
    """Prend en entrée la couche précédente pour faire le calcul sur tous les neurones précédents"""
    
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
    
    def __init__(self, last_layer:list, weights=None, bias=None):
        """
        Parameters
        ----------
        last_layer:list
            Liste des objets Node Précédents
        weights:list
            Liste des weights pour chaques input
        bias:float
            Nombre à additionner au résultat
        """
        n_last_node = len(last_layer)
        self.last_layer = last_layer
        self.weights = None
        self.bias = None 
        
        if not weights is None:
            if len(weights) == n_last_node:
                self.weights = weights
            else:
                raise AssertionError("Nombre de poids différent de celui des Nodes")
        
        if not bias is None:
            self.bias = bias
            
        if self.weights is None:
            self.weights = [random.uniform(self.min_weight, self.max_weight) for _ in range(n_last_node)]    
        if self.bias is None:
            self.bias = random.uniform(self.min_bias, self.max_bias)
            
        self.value = None 
            
    def calculate(self, activation_function="sigmoid"):
        """Calcule la nouvelle valeur du Noeud avec les valeurs des noeuds précédents"""
        if not activation_function in self.activation_functions:
            raise AssertionError("La fonction d'activation donnée n'existe pas")    
        f = self.activation_functions[activation_function]
        
        # Somme de la valeur * le poids de chaque noeuds + le biais
        res = 0
        
        for idx, node in enumerate(self.last_layer):
            w = self.weights[idx]
            res += node.value*w
        
        res = f(res+self.bias)
        self.value = res
        return res
            
class InputNode:
    """Classe qui contient les valeurs input"""
    def __init__(self, value):
        self.value = value
       
class NeuralNetwork:
    """Contient les couches contenant les noeuds"""

    variation_base = 0.1

    def __init__(self, n_layers,hidden_size, n_input, n_output, hidden_activation_function:str="relu", output_activation_function="linear"):
        """
        Parameters
        ----------
        n_layers:int
            Nombre de couches
        hidden_size:int
            Nombre de noeuds pour chaques couches
        n_input:int
            Taille de l'input
        n_output:int
            Taille de l'output
        """
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

        self.input_layer = [InputNode(0) for _ in range(n_input)]

        self.layers = [] # Liste de listes de noeuds
        
        for i in range(n_layers):
            layer_nodes = []
            for _ in range(hidden_size): # Nombre de noeuds dans une couche
                if i == 0:
                    layer_nodes.append(Node(self.input_layer))
                else:
                    layer_nodes.append(Node(self.layers[-1]))
            self.layers.append(layer_nodes)
        
        if n_layers != 0:
            self.output_layer = [Node(self.layers[-1]) for _ in range(n_output)]
        else:
            self.output_layer = [Node(self.input_layer) for _ in range(n_output)]

    def prediction(self, inputs:list=[]):
        """
        Calcule les outputs en fonction des valeurs en input

        Parameters
        ----------
        inputs:list
            Valeurs numériques
        """

        for n in inputs:
            if type(n) != int and type(n) != float:
                raise AssertionError(f"{type(n)} trouvé au lieu de {int}")
        
        if len(inputs) != len(self.input_layer):
            raise AssertionError(f"{len(inputs)} inputs trouvés au lieu de {len(self.input_layer)}")
        
        # Initier les valeurs dans `input_layer`
        for idx, val in enumerate(inputs):
            self.input_layer[idx].value = val

        # Calculer les valeurs pour chaques couches
        for idx, layer in enumerate(self.layers):
            for node in layer:
                node.calculate(activation_function=self.hidden_activation_function)

        # Calculer les valeurs `output`
        for node in self.output_layer:
            node.calculate(activation_function=self.output_activation_function)

        return [node.value for node in self.output_layer]
    
    def mutate(self, error, mutation_base=None):
        """
        Modifie les poids et les biais des noeuds aléatoirement dans une range qui varie en fonction de l'erreur

        Parameters
        ----------
        error:float
            Coefficient d'erreur, peut être négatif
        """
        if mutation_base is None:
            base_mutation = self.variation_base
        else:
            base_mutation = mutation_base
        cur_variation = base_mutation * min(abs(error), 1.0)
        
        # Modifier pour toutes les couches
        for idx, layer in enumerate(self.layers):
            for node in layer:
                node.weights = [node.weights[i] + random.uniform(-cur_variation, cur_variation) for i in range(len(node.weights))]
                node.bias = node.bias + random.uniform(-cur_variation, cur_variation)

        # Modifier pour l'output
        for node in self.output_layer:
            node.weights = [node.weights[i] + random.uniform(-cur_variation, cur_variation) for i in range(len(node.weights))]
            node.bias = node.bias + random.uniform(-cur_variation, cur_variation)

    def export(self):
        """
        Retourne les valeurs des Noeuds sous forme de liste
        
        Returns
        -------
        list
            Liste des couches sous forme de listes de noeuds sous forme de liste contenant les poids et le biais
            On ignore la couche input
        """
        
        export = []
        
        for layer in self.layers:
            layer_export = []
            for node in layer:
                layer_export.append([node.weights, node.bias])
            export.append(layer_export)
        

class GeneticAi:

    error_functions = {
        "mse": mse,
        "erreur_relative": erreur_relative,
        "bce": bce,
        "erreur_binaire": binary_error,
    }

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

    def train(self, training_data:dict, epochs:int, mutation_base:float=0.1):
        """
        Calcule la précision de chaque Réseau et garde les meilleurs pour les modifier epochs fois

        Parameters
        ----------
        training_data:dict
            Dictionnaire sous la forme {input:tuple, outputs:tuple}
        epochs:int
            Nombre de répétition
        base_mutation:float
            Variation maximum des valeurs
            
        Returns
        -------
        list
            Liste des meilleurs erreurs
        """
        errors = []
        
        for i in range(epochs):
            
            # Faire une liste triés du meilleur au pire et ne garder que les 1/10
            precisions = {nn: self.get_precision(nn, training_data) for nn in self.population}
            precisions_sorted = sorted(precisions.items(), key=lambda x: x[1])
            n_best = max(1, self.population_size // 5)
            best_networks = [nn for nn, _ in precisions_sorted[:n_best]]
            best = best_networks[0]
            err  = precisions[best]
            errors.append(err)
            print(f"Epoch {i+1}/{epochs} — meilleure erreur = {err:.4f}")

            # Recréer la population avec les 1/10 qui sont les mêmes et les autres sont des dérivés des 1/10 meilleurs
            best_index = 0 # Index de best_networks

            for idx, nn in enumerate(self.population):
                if idx < n_best:
                    self.population[idx] = best_networks[idx]
                    continue

                error = precisions[nn]
                copy(best_networks[best_index], nn)

                nn.mutate(error, mutation_base)
            
                best_index += 1

                if best_index == n_best:
                    best_index = 0
        
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
            results += self.error(neural_network.prediction(input_), output) # Calculer erreur relative

        return results / len(training_data)

if __name__ == "__main__":
    pass
