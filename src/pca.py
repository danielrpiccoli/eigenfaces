"""
PCA via autovalores/autovetores de A^T A.

Como o número de pixels (n) é muito maior que o número de amostras (m),
é mais eficiente calcular os autovetores de A^T A (m x m) e depois
recuperar os autovetores de A A^T (as eigenfaces de fato) através da
relação:

    Se A^T A v = lambda v, então A A^T (Av) = lambda (Av)

ou seja, se v é autovetor de A^T A com autovalor lambda, então Av é
autovetor de A A^T com o mesmo autovalor lambda (normalizando depois).
"""

import numpy as np


def calcular_pca(A, n_componentes=None):
    """
    Args:
        A: matriz (n_pixels x n_amostras), já centralizada
        n_componentes: quantos componentes principais retornar.
           Se None, retorna todos os componentes possíveis (até n_amostras - 1).

    Returns:
        autovetores: matriz (n_pixels x n_componentes), cada coluna é uma eigenface
        autovalores: vetor (n_componentes,) com os autovalores associados,
           em ordem decrescente
    """
    n_amostras = A.shape[1]

    if n_componentes is None:
        n_componentes = n_amostras

    # matriz pequena (n_amostras x n_amostras) em vez da grande (n_pixels x n_pixels)
    M = (A.T @ A) / (n_amostras - 1)

    autovalores, autovetores_M = np.linalg.eigh(M)

    # eigh retorna em ordem crescente -> inverte para pegar os maiores primeiro
    ordem = np.argsort(autovalores)[::-1]
    autovalores = autovalores[ordem]
    autovetores_M = autovetores_M[:, ordem]

    # recuperar autovetores de A A^T (eigenfaces) a partir dos autovetores de A^T A
    autovetores = A @ autovetores_M

    # normalizar cada eigenface para ter norma 1
    normas = np.linalg.norm(autovetores, axis=0)
    normas[normas == 0] = 1  # evita divisão por zero
    autovetores = autovetores / normas

    n_componentes = min(n_componentes, autovetores.shape[1])
    return autovetores[:, :n_componentes], autovalores[:n_componentes]


def variancia_explicada_acumulada(autovalores):
    """
    Dado o array de autovalores (em ordem decrescente, idealmente TODOS os
    autovalores possíveis, não só os truncados), retorna a fração acumulada
    da variância total explicada por cada número crescente de componentes.

    Ex: retorno[k-1] = fração da variância explicada usando os k primeiros
    componentes principais.
    """
    autovalores = np.clip(autovalores, a_min=0, a_max=None)  # evita negativos por erro numérico
    total = autovalores.sum()
    if total == 0:
        return np.zeros_like(autovalores)
    return np.cumsum(autovalores) / total


def n_componentes_para_variancia(autovalores, alvo):
    """
    Retorna o menor número de componentes necessário para explicar pelo
    menos 'alvo' (ex: 0.95 para 95%) da variância total.
    """
    acumulada = variancia_explicada_acumulada(autovalores)
    indices = np.where(acumulada >= alvo)[0]
    if len(indices) == 0:
        return len(autovalores)
    return int(indices[0]) + 1  # +1 porque índice 0 corresponde a 1 componente


def projetar(A, media, autovetores):
    """Projeta as colunas de A (já centralizadas por 'media') no espaço reduzido."""
    A_centralizada = A - media[:, None]
    return autovetores.T @ A_centralizada  # (n_componentes x n_amostras)


def reconstruir(coeficientes, media, autovetores):
    """Reconstrói as imagens originais a partir dos coeficientes projetados."""
    return autovetores @ coeficientes + media[:, None]
