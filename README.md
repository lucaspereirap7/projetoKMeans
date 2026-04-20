# Trabalho Prático 02 - Clusterização K-means (Inteligência Artificial)

Este projeto implementa o algoritmo de agrupamento **K-means** para análise do conjunto de dados **Iris**. O objetivo é comparar uma implementação manual ("hardcore"), desenvolvida do zero, com a implementação otimizada da biblioteca **Scikit-Learn**, avaliando a qualidade dos clusters e visualizando os resultados através de **PCA**.

---
**Trabalho desenvolvido pelos alunos:**  
**Renan Augusto da Silva** e **Lucas de Oliveira Pereira**  
**Curso:** Ciência da Computação - UFLA  
**Disciplina:** GCC 128 - Inteligência Artificial  
---

## Estrutura de Arquivos

*   **executar_clusterizacao.py**: Script principal que integra todos os módulos, realiza a normalização dos dados, executa os experimentos para k = 3 e k = 5 e gera o resumo final.
*   **kmeans_manual.py**: Implementação do algoritmo K-means do zero, contendo a lógica de inicialização de centróides, cálculo de distância euclidiana, atribuição de clusters e atualização iterativa.
*   **kmeans_biblioteca.py**: Módulo que utiliza a classe KMeans do Scikit-Learn para fins de comparação e validação.
*   **avaliacao.py**: Contém funções para cálculo do Silhouette Score e geração de Matrizes de Confusão (comparando clusters com as classes reais).
*   **visualizacao_pca.py**: Implementa a redução de dimensionalidade com PCA para visualização dos agrupamentos e centróides em espaços 1D e 2D.
*   **iris.csv**: Base de dados Iris utilizada como referência para os experimentos.

## Requisitos e Instalação

Certifique-se de ter o Python instalado (versão 3.x). As bibliotecas necessárias podem ser instaladas via terminal com o comando abaixo:

    pip install numpy pandas matplotlib seaborn scikit-learn

## Como Executar

Para rodar o projeto completo e visualizar todos os gráficos e métricas, execute no terminal:

    python executar_clusterizacao.py

## Funcionalidades e Resultados

Ao ser executado, o projeto realiza:
1.  **Clusterização Manual vs. Biblioteca**: Compara os dois métodos para k=3 e k=5.
2.  **Métricas de Qualidade**: Exibe o Silhouette Score de cada modelo (quanto mais próximo de 1, melhor a separação).
3.  **Medição de Desempenho**: Calcula o tempo de execução de cada abordagem utilizando time.perf_counter().
4.  **Visualização PCA**: Gera gráficos em 1 e 2 componentes para o melhor valor de K encontrado, plotando os pontos coloridos e seus respectivos centróides (marcados com 'X').

## Análise Resumida

*   **Melhor K**: O valor k = 3 apresentou o maior Silhouette Score, sendo consistente com a estrutura natural da base Iris (3 espécies).
*   **Implementação**: A versão manual atingiu métricas de qualidade muito próximas à versão da biblioteca, validando a implementação lógica do algoritmo.
*   **Desempenho**: O Scikit-Learn demonstrou maior velocidade de processamento devido às otimizações internas e técnicas de inicialização avançadas (como k-means++).
*   **Visualização**: O PCA permitiu observar que a espécie Setosa forma um grupo bem isolado, enquanto Versicolor e Virginica possuem zonas de sobreposição.

## Link da Apresentação

- https://youtu.be/-1woGDFHeV4