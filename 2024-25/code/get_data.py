# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def f(x, w):
    res = 0
    for i in range(len(w)):
        res = res + w[i] * x**i
    return res


a, b = 0, 1
sigma = 0.1

x = np.linspace(a, b, 1000)
alpha_true = np.array([0, 0, 4, 0, -3])

y = f(x, alpha_true)
y_delta = y + sigma * np.random.normal(0, 1, y.shape)

data = np.array([x, y_delta])
df = pd.DataFrame(
    data.T,
    columns=("x", "y"),
)
df.to_csv("./data/poly_regression_large.csv")


# %%
# SOLUZIONE
def vandermonde(x, d):
    r"""
    Preso in input un numpy array "x" di lunghezza (n, ) contentente i dati, e un valore intero "d" rappresentante il grado del polinomio,
    costruisce e ritorna la matrice di vandermonde X di grado d, associata a x.

    Parameters:
    x (ndarray): Il vettore dei dati di input.
    d (int): Il grado massimo del polinomio.

    Returns:
    X (ndarray): La matrice di Vandermonde di grado "d", associata ad x.
    """
    n = x.shape[0]

    # Inizializzo la matrice di Vandermonde con shape (n, d+1)
    X = np.zeros((n, d + 1))

    # Costruisco la matrice di Vandermonde
    for i in range(d + 1):
        X[:, i] = x**i
    return X


# Costruisco vandermonde
X = vandermonde(x, d=8)

# Troviamo la matrice L tale che X^T X = L L^T
L = np.linalg.cholesky(X.T @ X)

###### Risolviamo il primo sistema:
# Calcoliamo il termine noto X^T y
Xty = X.T @ y_delta

# Troviamo z
z = np.linalg.solve(L, Xty)

###### Risolviamo il secondo sistema:
# Troviamo alpha
alpha_chol = np.linalg.solve(L.T, z)


plt.plot(x, y)
# plt.plot(x, y_delta, "o")
plt.plot(x, f(x, alpha_chol), "-")
plt.grid()
plt.legend(["True", "Data", "Chol"])
plt.show()

# %%
