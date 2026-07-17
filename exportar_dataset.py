import numpy as np
from sklearn.datasets import fetch_olivetti_faces

dados = fetch_olivetti_faces()  # usa o cache local, não baixa de novo
np.savez_compressed("olivetti_faces.npz", imagens=dados.images, rotulos=dados.target)
print("Arquivo gerado: olivetti_faces.npz")
