"""
Classificação simples por vizinho mais próximo no espaço reduzido (eigenfaces).

Separa os dados em treino/teste, projeta ambos no espaço de componentes principais
e classifica cada rosto de teste pela distância euclidiana ao rosto de treino mais
próximo no espaço reduzido.
"""

import numpy as np
import matplotlib.pyplot as plt
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

    # calcula todos os componentes possíveis uma vez só; depois fatiamos
    # para testar diferentes números de componentes sem recalcular o PCA
    n_max = min(A_treino.shape[1] - 1, 150)  # limite razoável para não pesar demais
    autovetores_completos, _ = calcular_pca(A_treino_centralizada, n_componentes=n_max)

    lista_ranks = [1, 2, 5, 10, 20, 30, 50, 75, 100, n_max]
    lista_ranks = sorted(set(r for r in lista_ranks if r <= n_max))

    precisoes = []
    for r in lista_ranks:
        autovetores = autovetores_completos[:, :r]
        coef_treino = projetar(A_treino, media, autovetores)
        coef_teste = projetar(A_teste, media, autovetores)

        predicoes = classificar_vizinho_mais_proximo(coef_treino, rotulos_treino, coef_teste)
        precisao = np.mean(predicoes == rotulos_teste)
        precisoes.append(precisao)
        print(f"Precisão com {r} componentes: {precisao:.2%}")

    melhor_indice = int(np.argmax(precisoes))
    print(f"\nMelhor resultado: {precisoes[melhor_indice]:.2%} "
          f"com {lista_ranks[melhor_indice]} componentes")

    plt.figure(figsize=(6, 4))
    plt.plot(lista_ranks, precisoes, marker="o")
    plt.xlabel("Número de componentes (rank)")
    plt.ylabel("Precisão")
    plt.title("Precisão da classificação vs. número de componentes")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("../resultados/precisao_vs_rank.png", dpi=150)
    print("Figura salva em ../resultados/precisao_vs_rank.png")
    plt.show()


if __name__ == "__main__":
    main()
