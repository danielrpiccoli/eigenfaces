"""
Matriz de confusão para a classificação por vizinho mais próximo no espaço
de eigenfaces. Mostra quais pessoas o modelo mais confunde entre si.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from carregar_dados import carregar_matriz_rostos
from pca import calcular_pca, projetar
from reconhecer import classificar_vizinho_mais_proximo


def plotar_matriz_confusao(matriz, n_classes, caminho_saida=None):
    fig, eixo = plt.subplots(figsize=(9, 8))
    im = eixo.imshow(matriz, cmap="viridis")
    eixo.set_xlabel("Pessoa prevista")
    eixo.set_ylabel("Pessoa real")
    eixo.set_title("Matriz de confusão — reconhecimento facial via eigenfaces")
    eixo.set_xticks(range(0, n_classes, 5))
    eixo.set_yticks(range(0, n_classes, 5))
    plt.colorbar(im, ax=eixo, label="Número de fotos")
    plt.tight_layout()
    if caminho_saida:
        plt.savefig(caminho_saida, dpi=150)
        print(f"Figura salva em {caminho_saida}")
    plt.show()


def listar_maiores_confusoes(matriz, top_n=10):
    """
    Retorna os pares (pessoa_real, pessoa_prevista, contagem) fora da
    diagonal com maior número de confusões, ordenados do maior para o menor.
    """
    n_classes = matriz.shape[0]
    confusoes = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and matriz[i, j] > 0:
                confusoes.append((i, j, matriz[i, j]))
    confusoes.sort(key=lambda x: x[2], reverse=True)
    return confusoes[:top_n]


def main():
    A, rotulos, altura, largura = carregar_matriz_rostos()
    n_classes = len(set(rotulos))

    indices = np.arange(A.shape[1])
    idx_treino, idx_teste = train_test_split(
        indices, test_size=0.25, stratify=rotulos, random_state=42
    )

    A_treino, A_teste = A[:, idx_treino], A[:, idx_teste]
    rotulos_treino, rotulos_teste = rotulos[idx_treino], rotulos[idx_teste]

    media = A_treino.mean(axis=1)
    A_treino_centralizada = A_treino - media[:, None]

    n_componentes = 50
    autovetores, _ = calcular_pca(A_treino_centralizada, n_componentes=n_componentes)

    coef_treino = projetar(A_treino, media, autovetores)
    coef_teste = projetar(A_teste, media, autovetores)

    previsoes = classificar_vizinho_mais_proximo(coef_treino, rotulos_treino, coef_teste)

    precisao = np.mean(previsoes == rotulos_teste)
    print(f"Precisão com {n_componentes} componentes: {precisao:.2%}")
    print()

    matriz = confusion_matrix(rotulos_teste, previsoes, labels=range(n_classes))

    plotar_matriz_confusao(matriz, n_classes,
                            caminho_saida="../resultados/matriz_confusao.png")

    maiores_confusoes = listar_maiores_confusoes(matriz)
    if maiores_confusoes:
        print("Maiores confusões (pessoa real -> pessoa prevista : quantidade):")
        for real, prevista, contagem in maiores_confusoes:
            print(f"  Pessoa {real:2d} -> Pessoa {prevista:2d} : {contagem}x")
    else:
        print("Nenhuma confusão fora da diagonal — o modelo acertou tudo no conjunto de teste.")


if __name__ == "__main__":
    main()

