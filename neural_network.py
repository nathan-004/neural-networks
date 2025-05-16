import random
from math import exp

# -----------------------Activation Functions-------------------------
def relu(z):
    """Rectified Linear Units (ReLU)"""
    return max(0, z)

def sigmoid(z):
    """Sigmoid activation"""
    return 1/(1+exp(-1*z))

class Node:
    """Prend en entrée la couche précédente pour faire le calcul sur tous les neurones précédents"""
    
    max_weight = 1
    min_weight = 0.15
    min_bias = 1
    max_bias = 5
    activation_functions = {
        "relu": relu,
        "sigmoid": sigmoid,
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
                AssertionError("Nombre de poids différent de celui des Nodes")
        
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
            AssertionError("La fonction d'activation donnée n'existe pas")    
        f = self.activation_functions[activation_function]
        
        # Somme de la valeur * le poids de chaque noeuds + le biais
        res = 0
        
        for idx, node in enumerate(self.last_layer):
            w = self.weights[idx]
            res += f(node.value*w)
        
        res += self.bias
        self.value = res
        return res
            
class InputNode:
    """Classe qui contient les valeurs input"""
    def __init__(self, value):
        self.value = value
       
class NeuralNetwork:
    """Contient les couches contenant les noeuds"""

    variation_base = 2

    def __init__(self, n_layers,hidden_size, n_input, n_output):
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
        
        self.output_layer = [Node(self.layers[-1]) for _ in range(n_output)]

    def prediction(self, inputs:list=[]):
        """
        Calcule les outputs en fonction des valeurs en input

        Parameters
        ----------
        inputs:list
            Valeurs numériques
        """

        for n in inputs:
            if type(n) != int:
                raise AssertionError(f"{type(n)} trouvé au lieu de {int}")
        
        if len(inputs) != len(self.input_layer):
            raise AssertionError(f"{len(inputs)} inputs trouvés au lieu de {len(self.input_layer)}")
        
        # Initier les valeurs dans `input_layer`
        for idx, val in enumerate(inputs):
            self.input_layer[idx].value = val

        # Calculer les valeurs pour chaques couches
        for idx, layer in enumerate(self.layers):
            for node in layer:
                node.calculate()

        # Calculer les valeurs `output`
        for node in self.output_layer:
            node.calculate()

        return node.value
    
    def mutate(self, error):
        """
        Modifie les poids et les biais des noeuds aléatoirement dans une range qui varie en fonction de l'erreur

        Parameters
        ----------
        error:float
            Coefficient d'erreur, peut être négatif
        """

        cur_variation = self.variation_base*abs(error)
        
        # Modifier pour toutes les couches
        for idx, layer in enumerate(self.layers):
            for node in layer:
                node.weights = [node.weights[i] + random.uniform(-cur_variation, cur_variation) for i in range(len(node.weigths))]
                node.bias = node.bias + random.uniform(-cur_variation, cur_variation)

        # Modifier pour l'output
        for node in self.output_layer:
            node.weights = [node.weights[i] + random.uniform(-cur_variation, cur_variation) for i in range(len(node.weigths))]
            node.bias = node.bias + random.uniform(-cur_variation, cur_variation)

class GeneticAi:

    def __init__(self, population_size, n_layers,hidden_size, n_input, n_output):
        """
        Parameters
        ----------
        epochs:int
            Nombre de répétition
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

        self.population = [NeuralNetwork(n_layers, hidden_size, n_input, n_output) for _ in range(population_size)] # Liste des réseaux de Neurones

    def train(self, training_data:dict, epochs):
        """
        Calcule la précision de chaque Réseau et garde les meilleurs pour les modifier epochs fois

        Parameters
        ----------
        training_data:dict
            Dictionnaire sous la forme {input:tuple, outputs:tuple}
        """

        pass
    
if __name__ == "__main__":
    nn = NeuralNetwork(5, 3, 1, 1)
    print(nn.prediction([1]))
    nn.mutate(0.6)
