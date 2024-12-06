# UrbanReleaf

## Análise e Visualização de Regressão para Dados Geoespaciais

Este projeto utiliza **técnicas de regressão** para prever a temperatura média com base em variáveis derivadas de imagens geoespaciais, como NDVI (Vegetação), LST (Temperatura da Superfície), e SWIR (Umidade). O código também inclui visualizações para analisar o desempenho dos modelos, a importância das features e a visualização dos dados geoespaciais.

## Estrutura do Projeto

- **`main.py`**: Realiza o processamento dos dados, aplica diferentes modelos de regressão e avalia o desempenho.
- **`testmodel.py`**: Executa previsões de LST utilizando modelos treinados e gera visualizações das previsões.
- **`tiff_viewer.py`**: Visualiza arquivos `.tiff` e salva as visualizações como imagens `.png` para facilitar a inspeção dos dados geoespaciais.

---

## 1. `main.py`

### Funcionalidades:
1. **Carregamento de Dados**:
   - Utiliza a biblioteca `rasterio` para processar arquivos `.tiff` contendo matrizes de NDVI, LST (Temperatura), e SWIR.
   - Vetoriza as matrizes 2D em vetores 1D para alimentar os modelos de regressão.
   
2. **Área de Interesse e Intervalo de Tempo**:
   - **Área de São Paulo**: A análise abrange uma região de São Paulo.
   - **Intervalo de Tempo**: Os dados foram coletados ao longo de um ano inteiro.

3. **Pré-processamento dos Dados**:
   - **Escalonamento das Features**: Aplica `StandardScaler` para normalizar os dados, melhorando o desempenho dos modelos.
   - **Tratamento de Outliers**: Remove pontos de dados onde o NDVI está além de 3 desvios padrão da média para mitigar o impacto de outliers.

4. **Modelos de Regressão**:
   - **Linear Regression**: Modelo básico para identificar relações lineares.
   - **SGDRegressor**: Utiliza gradiente descendente estocástico para otimização, adequado para grandes conjuntos de dados.

5. **Avaliação do Desempenho**:
   - Calcula métricas como **Mean Squared Error (MSE)** e **R² Score** para cada modelo.

6. **Integração com Visualizações**:
   - Passa os resultados para as funções em `testmodel.py` para criar gráficos de desempenho e análise.

### Como Executar:
1. **Configuração Inicial**:
   - Certifique-se de que as variáveis de ambiente estejam configuradas corretamente no arquivo `.env`:
     ```env
     INSTANCE_ID=your_instance_id
     CLIENT_ID=your_client_id
     CLIENT_SECRET=your_client_secret
     ```

2. **Preparação dos Dados**:
   - O script está configurado para a área de São Paulo e o intervalo de tempo para tres anos.

3. **Execução do Script**:
   - Execute o script principal:
     ```bash
     python main.py
     ```

4. **Resultados**:
   - Os modelos treinados serão salvos na pasta `results/models/`.
   - Visualizações de desempenho e importâncias das features serão geradas conforme configurado no código.

---

## 2. `testmodel.py`

### Funcionalidades:
1. **Carregamento de Modelos e Scalers**:
   - Carrega os modelos treinados (`linear_regression_model.joblib` e `sgd_regressor_model.joblib`) e os objetos `StandardScaler` para normalização dos dados.
   
2. **Pré-processamento dos Dados**:
   - Processa arquivos `.tiff` de NDVI para preparação dos dados de entrada para os modelos de previsão.
   
3. **Previsão de LST**:
   - Utiliza os modelos treinados para prever a Temperatura da Superfície (LST) a partir dos dados de NDVI escalonados.
   
4. **Salvar e Visualizar Predições**:
   - Salva as predições de LST como arquivos `.tiff` e gera visualizações das predições em formato `.png` para análise.

### Como Executar:
1. **Configuração Inicial**:
   - Certifique-se de que os modelos treinados e os scalers estejam salvos nas pastas `results/models/` e `results/scalers/` respectivamente.
   
2. **Preparação dos Dados de Teste**:
   - Coloque os arquivos `.tiff` de NDVI na pasta `testdata/`. O script buscará automaticamente o arquivo disponível.
   
3. **Execução do Script**:
   - Execute o script de predição:
     ```bash
     python testmodel.py
     ```
   
4. **Resultados**:
   - As predições de LST serão salvas na pasta `results/predictions/` como arquivos `.tif`.
   - As visualizações das predições serão salvas na mesma pasta como arquivos `.png`.

---

## 3. `tiff_viewer.py`

### Funcionalidades:
1. **Visualização de Arquivos `.tiff`**:
   - Utiliza a biblioteca `rasterio` para ler arquivos geoespaciais no formato `.tiff`.
   - Gera visualizações das bandas individuais usando `matplotlib` e salva as imagens como `.png` na pasta especificada.

2. **Organização das Visualizações**:
   - As imagens geradas são salvas na pasta `results/tiff_viewer/`, facilitando a inspeção e análise visual dos dados geoespaciais.

### Como Executar:
1. **Preparação dos Dados**:
   - Certifique-se de que os arquivos `.tiff` estejam na pasta `data/` com os nomes correspondentes:
     - `NDVI_10m_20211231_SaoPaulo.tif`
     - `LST_100m_20211231_SaoPaulo.tif`
     - `SWIR_20m_20211231_SaoPaulo.tif`
   
2. **Execução do Script**:
   - Execute o script de visualização:
     ```bash
     python tiff_viewer.py
     ```
   
3. **Resultados**:
   - As visualizações serão salvas na pasta `results/tiff_viewer/` como arquivos `.png`.

---

## Dependências

Certifique-se de ter as seguintes bibliotecas instaladas:

- `rasterio`
- `numpy`
- `scikit-learn`
- `sentinelhub`
- `joblib`
- `matplotlib`
- `python-dotenv`

Você pode instalar todas as dependências utilizando o `pip`:

```bash
pip install -r requirements.txt