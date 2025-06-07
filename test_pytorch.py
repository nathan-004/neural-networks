from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn as nn
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import GaussianMutation

# Charge les données
df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values.reshape(-1, 1)

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)


class DiabetesMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = DiabetesMLP()
N = sum(p.numel() for p in model.parameters())  # total nombre de poids/biais

def fitness_fn(chromosome: torch.Tensor) -> torch.Tensor:
    ptr = 0
    for param in model.parameters():
        n = param.numel()
        param.data = chromosome[ptr:ptr + n].view_as(param.data)
        ptr += n

    with torch.no_grad():
        preds = model(X_train)
        preds_binary = (preds > 0.5).float()
        acc = (preds_binary == y_train).float().mean()
    return acc  # on veut maximiser l'accuracy

problem = Problem(
    "max",
    fitness_fn,
    solution_length=N,
    bounds=(-1.0, 1.0),
    vectorized=False,
    device="cpu"
)

mutation = GaussianMutation(problem, stdev=0.1)  # standard deviation = 0.1

ga = GeneticAlgorithm(
    problem,
    popsize=200,
    operators=[mutation]  # ✅ obligatoire désormais
)

StdOutLogger(ga)

ga.run(300)  # 300 générations

print("Best accuracy found:", ga.best_solution().fitness)
