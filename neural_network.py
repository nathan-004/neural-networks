import random

# -----------------------Activation Functions-------------------------
def relu(z):
    """Rectified Linear Units (ReLU)"""
    return max(0, z)

def sigmoid(z):
    """Sigmoid activation"""
    return 1/(1+exp(-1*z))

class Node:
    """Prend en entrée la couche précédente pour faire le calcul sur tous les neurones précédents"""
    
    max_weight = 0.5
    min_weight = 0.5
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
        return res
            
class InputNode:
    """Classe qui contient les valeurs input"""
    def __init__(self, value):
        self.value = value
       
class NeuralNetwork:
    
    def __init__(self, n_layers,hidden_size, input_values, n_output):
        
        self.input_layer = [InputNode(value) for value in input_values]
        
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
