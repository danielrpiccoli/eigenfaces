"""
Classificação simples por vizinho mais próximo no espaço reduzido (eigenfaces).

Separa os dados em treino/teste, projeta ambos no espaço de componentes principais
e classifica cada rosto de teste pela distância euclidiana ao rosto de treino mais
próximo no espaço reduzido.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from carregar_dados import carregar_matriz_rostos
from pca import calcular_pca, projetar


def classificar_vizinho_mais_proximo(coef_treino, rotulos_treino, coef_teste):
    """
    Args:
        coef_treino: (n_componentes x n_treino)
        rotulos_treino: (n_treino,)
        coef_teste: (n_componentes x n_teste)

    Returns:
        predicoes: (n_teste,) rótulo previsto para cada amostra de teste
    """
    predicoes = []
    for i in range(coef_teste.shape[1]):
        vetor_teste = coef_teste[:, i]
        distancias = np.linalg.norm(coef_treino - vetor_teste[:, None], axis=0)
        indice_mais_proximo = np.argmin(distancias)
        predicoes.append(rotulos_treino[indice_mais_proximo])
    return np.array(predicoes)


def main():
    A, rotulos, altura, largura = carregar_matriz_rostos()

    indices = np.arange(A.shape[1])
    idx_treino, idx_teste = train_test_split(
        indices, test_size=0.25, stratify=rotulos, random_state=42
    )

    A_treino, A_teste = A[:, idx_treino], A[:, idx_teste]
    rotulos_treino, rotulos_teste = rotulos[idx_treino], rotulos[idx_teste]

    media = A_treino.mean(axis=1)
    A_treino_centralizada = A_treino - media[:, None]

    n_componentes = 50
    autovetores, _ = calcular_pca(A_treino_centralizada, n_componentes)

    coef_treino = projetar(A_treino, media, autovetores)
    coef_teste = projetar(A_teste, media, autovetores)

    predicoes = classificar_vizinho_mais_proximo(coef_treino, rotulos_treino, coef_teste)

    acuracia = np.mean(predicoes == rotulos_teste)
    print(f"Acurácia com {n_componentes} componentes: {acuracia:.2%}")


if __name__ == "__main__":
    main()

