"""
Teste de sanidade: compara a implementação de PCA via autovalores de A^T A
(pca.py) com a rota alternativa via SVD direta (np.linalg.svd).

As duas rotas devem chegar matematicamente no mesmo resultado:

    A = U Sigma V^T  (SVD completa/econômica)

    - As colunas de U são os autovetores de A A^T (as eigenfaces).
    - autovalor_i = valor_singular_i^2 / (n_amostras - 1)
      (o denominador aparece porque pca.py normaliza A^T A por n_amostras - 1,
      seguindo a convenção usual de covariância amostral)

Como autovetores só são definidos a menos de sinal (v e -v são ambos
autovetores válidos), a comparação direta usa valor absoluto ou compara
o subespaço gerado, não o vetor exato.
"""

import time
import numpy as np

from carregar_dados import carregar_matriz_rostos
from pca import calcular_pca


def calcular_eigenfaces_via_svd(A_centralizada):
    """
    Calcula eigenfaces e autovalores via SVD direta de A_centralizada.

    Returns:
        autovetores: (n_pixels x n_componentes), mesma convenção de pca.py
        autovalores: (n_componentes,) consistente com a normalização de pca.py
    """
    n_amostras = A_centralizada.shape[1]

    # SVD "econômica": evita calcular a matriz U completa (n_pixels x n_pixels),
    # calculando só as primeiras n_amostras colunas de U, que é tudo que existe
    # de informação não-trivial (pois rank(A) <= n_amostras - 1 após centralizar)
    U, valores_singulares, _ = np.linalg.svd(A_centralizada, full_matrices=False)

    autovalores = (valores_singulares ** 2) / (n_amostras - 1)
    return U, autovalores


def alinhar_sinais(autovetores_a, autovetores_b):
    """
    Autovetores são definidos a menos de sinal. Para comparar corretamente,
    inverte o sinal das colunas de autovetores_b que estiverem "opostas" às
    de autovetores_a (usando o sinal do produto interno entre elas).
    """
    sinais = np.sign(np.sum(autovetores_a * autovetores_b, axis=0))
    sinais[sinais == 0] = 1
    return autovetores_b * sinais


def main():
    A, rotulos, altura, largura = carregar_matriz_rostos()
    media = A.mean(axis=1)
    A_centralizada = A - media[:, None]

    n_comparar = 50  # número de componentes a comparar (dos até ~399 possíveis)

    # --- rota 1: pca.py (autovalores de A^T A) ---
    inicio = time.perf_counter()
    autovetores_pca, autovalores_pca = calcular_pca(A_centralizada, n_componentes=n_comparar)
    tempo_pca = time.perf_counter() - inicio

    # --- rota 2: SVD direta ---
    inicio = time.perf_counter()
    autovetores_svd, autovalores_svd = calcular_eigenfaces_via_svd(A_centralizada)
    tempo_svd = time.perf_counter() - inicio
    autovetores_svd = autovetores_svd[:, :n_comparar]
    autovalores_svd = autovalores_svd[:n_comparar]

    print(f"Tempo via autovalores de A^T A: {tempo_pca*1000:.2f} ms")
    print(f"Tempo via SVD direta:            {tempo_svd*1000:.2f} ms")
    print()

    # --- comparar autovalores (não têm ambiguidade de sinal) ---
    diferenca_autovalores = np.abs(autovalores_pca - autovalores_svd)
    print(f"Maior diferença absoluta entre autovalores: {diferenca_autovalores.max():.2e}")
    print(f"Diferença relativa média: "
          f"{np.mean(diferenca_autovalores / (autovalores_pca + 1e-12)):.2e}")
    print()

    # --- comparar autovetores (corrigindo ambiguidade de sinal) ---
    autovetores_svd_alinhados = alinhar_sinais(autovetores_pca, autovetores_svd)
    diferenca_autovetores = np.linalg.norm(autovetores_pca - autovetores_svd_alinhados, axis=0)
    print(f"Maior diferença de norma entre eigenfaces (após alinhar sinal): "
          f"{diferenca_autovetores.max():.2e}")
    print(f"Diferença média: {diferenca_autovetores.mean():.2e}")
    print()

    if diferenca_autovalores.max() < 1e-6 and diferenca_autovetores.max() < 1e-6:
        print("✓ As duas rotas concordam (diferenças na casa de erro numérico de ponto flutuante).")
    else:
        print("⚠ Diferença maior que o esperado — vale investigar.")


if __name__ == "__main__":
    main()

