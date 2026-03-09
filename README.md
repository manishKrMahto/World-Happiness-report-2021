## World Happiness Report 2021 – Project Overview

This repository contains an exploratory data analysis (EDA), modeling, and clustering project based on the **World Happiness Report 2021** dataset. The main analysis lives in `World_Happiness_Report_2021_notebook.ipynb`, and any supporting app logic is in `app.py` (if present).

### Dataset (essentials)

- **Rows**: 149 countries  
- **Key columns**:
  - `Country name`, `Regional indicator`
  - `Ladder score` (happiness)
  - `Logged GDP per capita`
  - `Social support`
  - `Healthy life expectancy`
  - `Freedom to make life choices`
  - `Generosity`

The notebook loads the CSV, checks shapes and basic stats (`df.shape`, `df.describe()`), and confirms the features used in the analysis are complete.

### Core Insights & Charts (from the notebook)

- **Correlation with happiness (Ladder score)**  
  - A correlation table for numeric columns shows `Ladder score` is **strongly positively correlated** with:
    - `Logged GDP per capita`
    - `Social support`
    - `Healthy life expectancy`
    - `Freedom to make life choices`
  - `Generosity` has a **near-zero** correlation with `Ladder score`.

- **Regional bar charts**  
  These charts are created with `sns.barplot` and show **regional averages**:
  - **Average Ladder Score by Region**: Western Europe and similar regions have the highest average happiness; Sub‑Saharan Africa and parts of South Asia are lower.
  - **Average Logged GDP per Capita by Region**: Regions with higher GDP per capita align with higher happiness.
  - **Average Social Support by Region**: Strong social support is associated with higher regional happiness.
  - **Average Healthy Life Expectancy by Region**: Regions with longer healthy life expectancy also show higher average `Ladder score`.
  - **Average Freedom to Make Life Choices by Region**: Regions with higher perceived freedom score higher on happiness.

- **Scatter charts: drivers vs happiness (colored by region)**  
  Built with `sns.scatterplot(..., hue='Regional indicator', s=100)`:
  - `Logged GDP per capita` vs `Ladder score`
  - `Social support` vs `Ladder score`
  - `Healthy life expectancy` vs `Ladder score`
  - `Freedom to make life choices` vs `Ladder score`

  These charts all show **clear positive relationships**: countries with higher values on these drivers cluster at the upper end of the happiness scale, and regional color‑coding highlights distinct regional patterns.

- **Model performance (regression)**  
  - **Linear Regression** predicting `Ladder score`:
    - Training R² ≈ **0.82**
    - Testing R² ≈ **0.73**
  - **Random Forest Regressor**:
    - Training R² ≈ **0.97**
    - Testing R² ≈ **0.69**

  The charts and metrics in the notebook illustrate that:
  - Linear regression already captures most of the signal and generalizes well.
  - Random forest overfits (very high train R², slightly worse test R²), bringing limited benefit here.

- **Clustering & elbow method (K‑Means)**  
  - **Cluster size bar chart**: `pd.DataFrame(y_means).value_counts().plot(kind='bar')` shows how many countries fall into each of the **4 clusters**, revealing dominant vs smaller happiness profiles.
  - **Elbow curve**: plotting **SSE vs number of clusters (k = 1…10)** shows a clear **elbow at k ≈ 4**, supporting the choice of 4 clusters.
  - **Clustered scatter charts**: grids of scatter plots of each key driver vs `Ladder score`, colored by **cluster label**, visualize how different combinations of GDP, support, health, and freedom produce distinct happiness clusters.

### How to Use This Repo

- **Install dependencies**:

```bash
pip install -r requirements.txt
```

- **Run the analysis notebook**:
  1. Open `World_Happiness_Report_2021_notebook.ipynb` in Jupyter / VS Code.
  2. Run the cells in order to:
     - Load and inspect the dataset.
     - Generate EDA charts and correlation insights.
     - Train and evaluate the regression models.
     - Fit K‑Means, plot the elbow curve, and visualize clusters.

- **Run the app (optional, if `app.py` exists)**:
  - As a plain script (e.g. Flask/FastAPI):

    ```bash
    python app.py
    ```

  - Or as a Streamlit app:

    ```bash
    streamlit run app.py
    ```

  See `app.py` for the exact framework and any configuration details.

