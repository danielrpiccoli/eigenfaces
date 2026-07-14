"""
PCA implementado via autovalores/autovetores de A^T A.

Quando o número de pixels (n) é muito maior que o número de amostras (m),
é mais eficiente calcular os autovetores de A^T A (m x m) e depois
recuperar os autovetores de A A^T através da
relação:

    Se A^T A v = lambda v, então A A^T (Av) = lambda (Av)

ou seja, se v é autovetor de A^T A com autovalor lambda, então Av é
autovetor de A A^T com o mesmo autovalor lambda (normalizando depois).
"""

import numpy as np


def calcular_pca(A, n_componentes):
    """
    Args:
        A: matriz (n_pixels x n_amostras), já deve estar centralizada
           (média de cada linha subtraída)
        n_componentes: quantos componentes principais retornar

    Returns:
        autovetores: matriz (n_pixels x n_componentes), cada coluna é uma eigenface
        autovalores: vetor (n_componentes,) com os autovalores associados
    """
    n_amostras = A.shape[1]

    # matriz pequena (n_amostras x n_amostras) em vez da grande (n_pixels x n_pixels)
    M = (A.T @ A) / (n_amostras - 1)

    autovalores, autovetores_M = np.linalg.eigh(M)

    # retorna em ordem crescente, inverte para pegar os maiores primeiro
    ordem = np.argsort(autovalores)[::-1]
    autovalores = autovalores[ordem]
    autovetores_M = autovetores_M[:, ordem]

    # recuperar autovetores de A A^T (eigenfaces) a partir dos autovetores de A^T A
    autovetores = A @ autovetores_M

    # normalizar cada eigenface para ter norma 1
    normas = np.linalg.norm(autovetores, axis=0)
    normas[normas == 0] = 1  # evita divisão por zero
    autovetores = autovetores / normas

    return autovetores[:, :n_componentes], autovalores[:n_componentes]


def projetar(A, media, autovetores):
    """Projeta as colunas de A já centralizadas por 'media' no espaço reduzido."""
    A_centralizada = A - media[:, None]
    return autovetores.T @ A_centralizada  # (n_componentes x n_amostras)


def reconstruir(coeficientes, media, autovetores):
    """Reconstrói as imagens originais a partir dos coeficientes projetados."""
    return autovetores @ coeficientes + media[:, None]

