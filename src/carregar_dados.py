"""
Baixa a base de dados Olivetti Faces e organiza como matriz A (pixels x amostras)
"""

import numpy as np 
from sklearn.datasets import fetch_olivetti_faces

def carregar_matriz_rostos():
    """
    Retorna
    A: Matriz (4096 x 400) onde cada coluna eh um rosto (imagem 64 x 64 vetoriazada)
    rotulos: Vetor (400, ) com o id da pessoa de cada coluna (0 a 39)
    altura, largura: Dimensões originais da imagem (64, 64)
    """
    dados = fetch_olivetti_faces()
    imagens = dados.images # shape: (400, 64, 64)
    rotulos = dados.target # shape: (400,)

    altura, largura = imagens.shape[1], imagens.shape[2]

    # Cada coluna de eh um rosto esticado em vetor de 4096 elementos
    A = imagens.reshape(imagens.shape[0], -1).T # shape: (4096, 400)

    return A, rotulos, altura, largura
if __name__ == "__main__":
    A, rotulos, altura, largura = carregar_matriz_rostos()
    print(f"Matriz A: {A.shape[0]} pixels x {A.shape[1]} amostras")
    print(f"Dimensao original das imagens: {altura}x{largura}")
    print(f"Numero de pesoas distintas: {len(set(rotulos))}")
