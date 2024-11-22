# UrbanReleaf

# Análise e Visualização de Regressão para Dados Geoespaciais

Este projeto utiliza **técnicas de regressão** para prever a temperatura média com base em variáveis derivadas de imagens geoespaciais, como NDVI (vegetação), LST (temperatura) e SWIR (umidade). O código também inclui visualizações para analisar o desempenho dos modelos e a importância das features.

## Estrutura do Projeto

- **`main.py`**: Realiza o processamento dos dados, aplica diferentes modelos de regressão e avalia o desempenho.
- **`graphs.py`**: Gera gráficos comparativos para facilitar a análise dos resultados dos modelos.

---

## 1. `main.py`

### Funcionalidades:
1. **Carregamento de Dados**:
   - Utiliza a biblioteca `rasterio` para processar arquivos `.tiff` contendo matrizes de NDVI, LST (temperatura), SWIR.
   - Vetoriza as matrizes 2D em vetores 1D para alimentar os modelos de regressão.

2. **Modelos de Regressão**:
   - **Linear Regression**: Modelo básico para identificar relações lineares.
   - **Ridge Regression**: Adiciona regularização para evitar overfitting.
   - **Random Forest Regressor**: Modelo não linear que captura padrões complexos.

3. **Avaliação do Desempenho**:
   - Calcula métricas como **Mean Squared Error (MSE)** e **R² Score** para cada modelo.
   - Permite comparar a eficiência das diferentes abordagens.

4. **Integração com Visualizações**:
   - Passa os resultados para as funções em `graphs.py` para criar gráficos de desempenho e análise.

### Como Executar:
1. Certifique-se de que os arquivos `.tiff` estejam na pasta `data`, localizada no mesmo diretório do código, com os seguintes nomes:
   - `ndvi_data.tiff`
   - `lst_data.tiff`
   - `swir_data.tiff`
2. Execute o script:
   ```bash
   python main.py