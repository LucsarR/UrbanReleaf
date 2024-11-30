# UrbanReleaf

## Análise e Visualização de Regressão para Dados Geoespaciais

Este projeto utiliza **técnicas de regressão** para prever a temperatura média com base em variáveis derivadas de imagens geoespaciais, como NDVI (Vegetação), LST (Temperatura da Superfície), e SWIR (Umidade). O código também inclui visualizações para analisar o desempenho dos modelos, a importância das features e a visualização dos dados geoespaciais.

## Estrutura do Projeto

- **`main.py`**: Realiza o processamento dos dados, aplica diferentes modelos de regressão e avalia o desempenho.
- **`graphs.py`**: Gera gráficos comparativos para facilitar a análise dos resultados dos modelos.
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
   - **Ridge Regression**: Adiciona regularização para evitar overfitting.
   - **Random Forest Regressor**: Modelo não linear que captura padrões complexos.

5. **Avaliação do Desempenho**:
   - Calcula métricas como **Mean Squared Error (MSE)** e **R² Score** para cada modelo.
   - Implementa validação cruzada (cross-validation) para obter estimativas mais robustas do desempenho dos modelos.
   - Gera visualizações de importância das features para modelos como Random Forest.

6. **Integração com Visualizações**:
   - Passa os resultados para as funções em `graphs.py` para criar gráficos de desempenho e análise.

### Como Executar:
1. **Configuração Inicial**:
   - Certifique-se de que as variáveis de ambiente estejam configuradas corretamente no arquivo `.env`:
     ```env
     INSTANCE_ID=your_instance_id
     CLIENT_ID=your_client_id
     CLIENT_SECRET=your_client_secret
     ```

2. **Preparação dos Dados**:
   - O script expandiu a área de São Paulo e estendeu o intervalo de tempo para um ano. Certifique-se de ter capacidade de armazenamento e processamento suficientes para lidar com o volume de dados aumentado.

3. **Execução do Script**:
   - Execute o script principal:
     ```bash
     python main.py
     ```

4. **Resultados**:
   - Os modelos treinados serão salvos na pasta `results/models/`.
   - Visualizações de desempenho e importâncias das features serão geradas conforme configurado no código.

---

## 2. `graphs.py`

### Funcionalidades:
1. **Plotagem de Desempenho dos Modelos**:
   - **MSE e R²**: Gera gráficos de barras comparando o Mean Squared Error e o R² Score de cada modelo.
   
2. **Comparação de Valores Reais vs. Previstos**:
   - Cria gráficos de dispersão para visualizar a relação entre os valores reais e previstos pelos modelos.
   
3. **Importância das Features**:
   - Exibe a importância das features para modelos que suportam essa funcionalidade, como o Random Forest Regressor.

### Como Utilizar:
As funções em `graphs.py` são chamadas automaticamente pelo `main.py` após a avaliação dos modelos. Certifique-se de que a biblioteca `matplotlib` e `seaborn` estejam instaladas para gerar as visualizações.

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
- `seaborn`
- `python-dotenv`

Você pode instalar todas as dependências utilizando o `pip`:

```bash
pip install -r requirements.txt