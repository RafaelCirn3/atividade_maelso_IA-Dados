# Projeto de Ciência de Dados com KNN

## Objetivo

Aplicar o fluxo completo de um projeto de Ciência de Dados, desde a preparação dos dados até a tunagem de hiperparâmetros, utilizando o algoritmo de Vizinhos Mais Próximos (K-Nearest Neighbors).

## 1. Formação dos Grupos e Escolha do Dataset

### Formação

- Grupos de até 3 integrantes.
- Apenas 1 envio é necessário.

### Fonte

- Kaggle: https://www.kaggle.com/datasets

### Requisito do Dataset

- Escolher um conjunto de dados para classificação.
- Preferir um dataset com quantidade razoável de atributos numéricos, para que o cálculo das distâncias seja relevante no KNN.

### Dataset escolhido

- Nome:
- Link:
- Variável alvo:
- Quantidade de linhas:
- Quantidade de atributos:
- Tipo das variáveis:

## 2. Etapa de Pré-processamento

Antes de executar o modelo, o grupo deve tratar os dados. A qualidade do KNN depende diretamente desta etapa.

### 2.1 Tratamento de Missing Values

- Identificar valores ausentes.
- Aplicar remoção ou imputação, conforme o caso.

Justificativa:

- O KNN não lida bem com valores ausentes, pois a distância entre observações depende de todos os atributos disponíveis.
- A imputação preserva mais dados quando a perda de linhas seria grande.

### 2.2 Codificação de Variáveis Categóricas

- Aplicar One-Hot Encoding ou Label Encoding, quando necessário.

Justificativa:

- O KNN trabalha com distância numérica, então variáveis categóricas precisam ser convertidas para valores numéricos.
- O One-Hot Encoding evita impor ordem artificial entre categorias.

### 2.3 Normalização ou Padronização

- Aplicar Min-Max Scaling ou StandardScaler.

Justificativa:

- O KNN é baseado em distância.
- Atributos com escalas maiores podem dominar o cálculo se os dados não forem reescalados.

### Comparações solicitadas

- Distribuição antes e depois da normalização.
- Comparação visual dos atributos mais relevantes, se possível.

### Técnicas aplicadas no projeto

- Missing values:
- Codificação categórica:
- Reescala:

## 3. Divisão Treino/Teste

- Separar os dados em 80% para treino e 20% para teste.
- Utilizar divisão estratificada para manter a proporção das classes.

### Configuração usada

- `test_size = 0.2`
- `stratify = y`
- `random_state =`

## 4. Implementação e Experimentação

O script deve testar diferentes combinações de hiperparâmetros para encontrar a melhor configuração do KNN.

### 4.1 Busca pelo K ideal

- Testar uma faixa de valores de K.
- Sugestão: de 1 até 15.

### 4.2 Métricas de distância

Testar, no mínimo, as quatro distâncias abaixo:

- Euclidiana: \( d(x, y) = \sqrt{\sum (x_i - y_i)^2} \)
- Manhattan: \( d(x, y) = \sum |x_i - y_i| \)
- Chebyshev: \( d(x, y) = \max |x_i - y_i| \)
- Minkowski: generalização das anteriores, com \( p \geq 3 \)

### Configuração no Scikit-Learn

- `metric = 'euclidean'`
- `metric = 'manhattan'`
- `metric = 'chebyshev'`
- `metric = 'minkowski'` com `metric_params = {'p': 3}`

### Estrutura da experimentação

Para cada valor de K e para cada métrica:

1. Treinar o modelo no conjunto de treino.
2. Avaliar no conjunto de teste.
3. Registrar a acurácia obtida.
4. Comparar os resultados em um gráfico.

## 5. Análise Exploratória

### Descrição do dataset

- Contexto do problema:
- O que a variável-alvo representa:
- Principais atributos:
- Distribuição das classes:

### Observações iniciais

- Há classes desbalanceadas?
- Existem atributos com escalas muito diferentes?
- Há variáveis categóricas que exigem codificação?

## 6. Resultados

### Gráfico de desempenho

- Comparar acurácia versus valor de K para cada métrica de distância.
- Apresentar um gráfico ou múltiplos gráficos.

### Tabela de resultados

| K | Euclidiana | Manhattan | Chebyshev | Minkowski |
|---|---:|---:|---:|---:|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |
| 6 |  |  |  |  |
| 7 |  |  |  |  |
| 8 |  |  |  |  |
| 9 |  |  |  |  |
| 10 |  |  |  |  |
| 11 |  |  |  |  |
| 12 |  |  |  |  |
| 13 |  |  |  |  |
| 14 |  |  |  |  |
| 15 |  |  |  |  |

## 7. Conclusão

- Melhor combinação encontrada de K + distância:
- Acurácia final no conjunto de teste:
- Houve sinais de overfitting com K muito baixo?
- Qual métrica apresentou o melhor desempenho?
- O pré-processamento foi decisivo para o resultado?

## 8. Referências

- Kaggle: https://www.kaggle.com/datasets
- Scikit-Learn: https://scikit-learn.org/

## 9. Observações finais

- Este arquivo pode ser usado como relatório ou como roteiro para o notebook.
- Se desejar, o próximo passo é transformar esta estrutura em um notebook com código e gráficos.