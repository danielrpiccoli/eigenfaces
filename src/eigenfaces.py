"""
Gera e visualiza as eigenfaces, além de mostrar reconstruções com rank crescente.
"""

import numpy as np
import matplotlib.pyplot as plt

from carregar_dados import carregar_matriz_rostos
from pca import (
    calcular_pca,
    projetar,
    reconstruir,
    variancia_explicada_acumulada,
    n_componentes_para_variancia,
)


def plotar_galeria(imagens, titulos, altura, largura, n_linhas=2, n_colunas=5,
                    caminho_saida=None):
    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(1.8 * n_colunas, 2.2 * n_linhas))
    for i, eixo in enumerate(eixos.flat):
        if i < len(imagens):
            eixo.imshow(imagens[i].reshape(altura, largura), cmap="gray")
            eixo.set_title(titulos[i], fontsize=9)
        eixo.axis("off")
    plt.tight_layout()
    if caminho_saida:
        plt.savefig(caminho_saida, dpi=150)
        print(f"Figura salva em {caminho_saida}")
    plt.show()


def main():
    A, rotulos, altura, largura = carregar_matriz_rostos()

    media = A.mean(axis=1)
    A_centralizada = A - media[:, None]

    # calcula TODOS os componentes possíveis de uma vez (até n_amostras - 1);
    # depois fatiamos conforme a necessidade em vez de recalcular o PCA várias vezes
    autovetores_completos, autovalores_completos = calcular_pca(A_centralizada, n_componentes=None)

    n_componentes = 50
    autovetores = autovetores_completos[:, :n_componentes]
    autovalores = autovalores_completos[:n_componentes]

    # visualizar as 10 primeiras eigenfaces
    titulos = [f"Eigenface {i+1}" for i in range(10)]
    plotar_galeria(autovetores[:, :10].T, titulos, altura, largura,
                    caminho_saida="../resultados/eigenfaces.png")

    # --- variância explicada acumulada ---
    acumulada = variancia_explicada_acumulada(autovalores_completos)

    for alvo in [0.90, 0.95, 0.99]:
        k = n_componentes_para_variancia(autovalores_completos, alvo)
        print(f"Componentes necessários para {alvo:.0%} da variância: {k}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(acumulada) + 1), acumulada, marker=".")
    plt.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="95%")
    plt.axhline(0.99, color="gray", linestyle=":", linewidth=1, label="99%")
    plt.xlabel("Número de componentes")
    plt.ylabel("Variância explicada acumulada")
    plt.title("Variância explicada vs. número de componentes")
    plt.legend()
    plt.grid(True)
    plt.savefig("../resultados/variancia_explicada.png", dpi=150)
    print("Figura salva em ../resultados/variancia_explicada.png")
    plt.show()

    # reconstrução com rank crescente para um rosto de exemplo
    indice_exemplo = 0
    ranks = [5, 20, 50]
    imagens_reconstruidas = [A[:, indice_exemplo]]  # original
    titulos_reconstrucao = ["Original"]

    for r in ranks:
        coef = projetar(A[:, [indice_exemplo]], media, autovetores_completos[:, :r])
        recon = reconstruir(coef, media, autovetores_completos[:, :r])
        imagens_reconstruidas.append(recon[:, 0])
        titulos_reconstrucao.append(f"Rank {r}")

    plotar_galeria(imagens_reconstruidas, titulos_reconstrucao, altura, largura,
                    n_linhas=1, n_colunas=len(imagens_reconstruidas),
                    caminho_saida="../resultados/reconstrucao.png")

    # erro de reconstrução (norma de Frobenius) vs número de componentes
    erros = []
    lista_ranks = list(range(1, n_componentes + 1, 2))
    for r in lista_ranks:
        # nota: 'projetar' já centraliza internamente (subtrai a média),
        # por isso passamos A (dados originais), não A_centralizada
        coef = projetar(A, media, autovetores_completos[:, :r])
        recon = reconstruir(coef, media, autovetores_completos[:, :r])
        erro = np.linalg.norm(A - recon, ord="fro")
        erros.append(erro)

    plt.figure(figsize=(6, 4))
    plt.plot(lista_ranks, erros, marker="o")
    plt.xlabel("Número de componentes (rank)")
    plt.ylabel("Erro de Frobenius")
    plt.title("Erro de reconstrução vs. número de componentes")
    plt.grid(True)
    plt.savefig("../resultados/erro_vs_rank.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

