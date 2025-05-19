# neural-networks
Entrainement aux réseaux de neurones.

## Résultats

Voici l'approximation de la fonction `sin(x)` avec cet entraînement :
```python
ai = GeneticAi(
population_size=100,
n_layers=3,
hidden_size=10,
n_input=1,
n_output=1,
erreur_calcul="mse",
)

ai.train(training_data, epochs=1000)
```
Sur des données d'entraînement allant de -10 à 10

![Graphique de l'approximation de sin](/images/sin_approximation.png)

## Ressources

https://medium.com/coinmonks/the-mathematics-of-neural-network-60a112dd3e05
