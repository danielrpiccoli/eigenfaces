# Reconhecimento Facial com Eigenfaces (PCA / SVD)

Projeto final de Modelos Aproximados / Computação Científica (UFRJ) — implementação
de reconhecimento facial simplificado usando a técnica clássica de **Eigenfaces**
(Turk & Pentland, 1991), aplicando **PCA** e **decomposição SVD** a partir dos
fundamentos de álgebra linear: autovalores, autovetores, ortogonalidade e
aproximação de posto baixo.

## Ideia geral

Cada rosto (imagem 64×64 pixels) é tratado como um vetor de 4096 números. A partir
de um conjunto de 400 rostos, o PCA encontra as direções de maior variação entre
eles — as **eigenfaces** — permitindo comprimir cada rosto para poucas dezenas de
coeficientes sem perder a informação relevante para identificação. A identificação
de uma pessoa nova é feita comparando, nesse espaço reduzido, a distância entre o
rosto de teste e todos os rostos de treino conhecidos.

## Destaques do projeto

- PCA implementado **do zero**, via autovalores de $A^TA$ (não usa
  `sklearn.decomposition.PCA` pronto) — com a matemática comentada no código.
- Comparação com **SVD direta** como teste de sanidade, incluindo análise do número
  de condição e discussão de estabilidade numérica.
- Classificação por vizinho mais próximo com validação treino/teste apropriada
  (sem vazamento de dados).
- Matriz de confusão e demonstração passo a passo de uma classificação individual.
- Notebook único e narrativo, pronto para rodar no Google Colab ou localmente.

## Dataset

**Olivetti Faces** — 40 pessoas, 10 fotos cada (400 imagens), 64×64 pixels em
escala de cinza. Carregado via `sklearn.datasets.fetch_olivetti_faces()`.

> **Nota:** o download desse dataset (hospedado no Figshare) pode falhar
> intermitentemente com erro `403 Forbidden` em ambientes de nuvem como o Google
> Colab — é um problema conhecido e recorrente do lado do Figshare, não do código
> deste projeto. Veja a seção [Solução de problemas](#solução-de-problemas) abaixo.

## Estrutura do repositório

```
eigenfaces/
├── README.md
├── requisitos.txt
├── .gitignore
├── dados/                          # dataset em cache (ignorado no git)
├── src/
│   ├── carregar_dados.py           # carrega o dataset Olivetti em matriz A
│   ├── pca.py                      # PCA via autovalores de AᵀA, projeção, reconstrução
│   ├── eigenfaces.py                # eigenfaces, reconstrução, variância explicada
│   ├── reconhecer.py                # classificação por vizinho mais próximo
│   ├── matriz_confusao.py           # matriz de confusão da classificação
│   └── comparar_svd.py              # teste de sanidade: PCA via AᵀA vs. SVD direta
├── notebooks/
│   └── eigenfaces_completo.ipynb    # notebook consolidado e narrativo (recomendado)
├── resultados/                      # gráficos e imagens gerados
└── relatorio/                       # relatório técnico completo do projeto
```

## Como rodar

### Opção 1 — Notebook (recomendado)

Abre `notebooks/eigenfaces_completo.ipynb` localmente (Jupyter/VS Code) ou faz
upload no [Google Colab](https://colab.research.google.com). O notebook é
autocontido — não depende dos arquivos em `src/`, roda do início ao fim sozinho.

### Opção 2 — Scripts individuais

```bash
python -m venv venv
source venv/bin/activate
pip install -r requisitos.txt

cd src
python carregar_dados.py        # testa o carregamento do dataset
python eigenfaces.py            # eigenfaces, reconstrução, variância explicada
python reconhecer.py            # classificação e acurácia vs. número de componentes
python matriz_confusao.py       # matriz de confusão
python comparar_svd.py          # comparação PCA vs. SVD direta
```

> No Arch Linux (e outras distros com Python gerenciado pelo sistema), use um venv
> como acima, ou adicione `--break-system-packages` ao `pip install` se preferir
> instalar globalmente.

## Resultados obtidos

| Métrica | Valor |
|---|---|
| Componentes para 95% da variância explicada | 38 |
| Componentes para 99% da variância explicada | 177 |
| Acurácia da classificação (50 componentes) | 94% |
| Erros na matriz de confusão | 6 de 100 fotos de teste |
| Diferença numérica PCA (via AᵀA) vs. SVD direta | ~10⁻⁶, consistente com a teoria |

Detalhamento completo — incluindo a fundamentação matemática, os bugs encontrados
durante o desenvolvimento e como foram corrigidos, e as respostas para as
perguntas mais prováveis de banca — está em [`relatorio/`](./relatorio).

## Limitações conhecidas

- O classificador sempre atribui a foto de teste a alguma das 40 pessoas do
  treino — não há rejeição de "pessoa desconhecida".
- Assume rostos já centralizados e padronizados (como no dataset Olivetti); fotos
  do mundo real exigiriam pré-processamento adicional (detecção e alinhamento
  facial).

## Solução de problemas

**`HTTPError: 403 Forbidden` ao rodar `fetch_olivetti_faces()` no Colab:**
é um bloqueio intermitente do Figshare a requisições vindas de datacenters/nuvem,
não um problema deste código. Soluções, em ordem de preferência:
1. Tentar novamente em alguns minutos (costuma ser temporário).
2. Rodar localmente, onde o dataset já pode estar em cache
   (`~/scikit_learn_data/`).
3. Exportar o dataset já baixado localmente para um `.npz`
   (`np.savez_compressed("olivetti_faces.npz", imagens=dados.images, rotulos=dados.target)`),
   fazer upload manual desse arquivo no Colab, e carregar via `np.load()` em vez de
   `fetch_olivetti_faces()` — elimina a dependência de rede externa.

## Referências

- Turk, M., & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of
  Cognitive Neuroscience.
- Dataset: [Olivetti Faces (AT&T)](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset)
