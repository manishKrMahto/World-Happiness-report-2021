## World Happiness Report 2021 – Analysis & Modeling

This repository contains an exploratory data analysis (EDA), predictive modeling, and clustering project based on the **World Happiness Report 2021** dataset. The main work is carried out in the notebook `World_Happiness_Report_2021_notebook.ipynb`, and any associated application logic (for example in `app.py`) is designed to reuse the same cleaned data and trained models.

### Dataset

- **Rows**: 149 countries  
- **Key columns** (among others):
  - `Country name`, `Regional indicator`
  - `Ladder score` (happiness score)
  - `Logged GDP per capita`
  - `Social support`
  - `Healthy life expectancy`
  - `Freedom to make life choices`
  - `Generosity`
  - `Perceptions of corruption`
  - Decomposition fields such as `Explained by: Log GDP per capita`, `Explained by: Social support`, etc.

The notebook loads the CSV file, inspects basic statistics (`df.describe()`), and verifies that the dataset is complete for the features used in modeling.

### Exploratory Data Analysis & Charts

The notebook performs several EDA steps and generates multiple charts:

- **Summary statistics**:
  - `Ladder score` ranges roughly from 2.5 to 7.8 with a mean around 5.5.
  - `Logged GDP per capita`, `Social support`, `Healthy life expectancy`, and `Freedom to make life choices` all show substantial variation across countries.
  - `Generosity` has both positive and negative values, indicating differences in perceived generosity across countries.

- **Correlation with happiness (Ladder score)**:
  - A correlation table shows that `Ladder score` is **strongly positively correlated** with:
    - `upperwhisker` and `lowerwhisker`
    - `Logged GDP per capita`
    - `Social support`
    - `Healthy life expectancy`
    - `Freedom to make life choices`
  - `Generosity` has a **very weak correlation** with `Ladder score`.

- **Regional bar charts**:
  - **Average Ladder Score by Region**: `sns.barplot` of `Ladder score` vs `Regional indicator` highlights that Western Europe and similar regions have the highest average happiness, while some other regions (e.g. Sub-Saharan Africa) are lower.
  - **Average Logged GDP per Capita by Region**: Shows that richer regions in terms of GDP per capita also tend to have higher happiness scores.
  - **Average Social Support by Region**: Regions with higher social support scores generally align with higher happiness.
  - **Average Healthy Life Expectancy by Region**: Regions with longer healthy life expectancy also show higher average `Ladder score`.
  - **Average Freedom to Make Life Choices by Region**: Regions where people perceive more freedom to make life choices show higher happiness.

- **Scatter plots of key drivers vs happiness (colored by region)**:
  - `Logged GDP per capita` vs `Ladder score`
  - `Social support` vs `Ladder score`
  - `Healthy life expectancy` vs `Ladder score`
  - `Freedom to make life choices` vs `Ladder score`

  In each case, the plots:
  - Use `sns.scatterplot` with `hue='Regional indicator'`.
  - Show clear **positive relationships** between each driver and `Ladder score`.
  - Reveal that regions with higher values of these drivers cluster at the higher end of the happiness scale.

- **Distribution view across variables**:
  - A combined bar/summary chart is used to visualize the distribution of several variables side by side with customized titles and axis labels, summarizing how the indicators differ in scale and spread.

Overall, the EDA and charts confirm that **economic prosperity (GDP), social support, health, and perceived freedom are key correlates of happiness**, while generosity has little direct linear relationship.

### Predictive Modeling

The notebook builds supervised models to predict `Ladder score`:

- **Feature preparation**:
  - A subset of numeric explanatory variables (including GDP, social support, health, freedom, etc.) is selected as `x`.
  - `Ladder score` is used as the target `y`.
  - Data is split using `train_test_split` with an 80/20 split.
  - In one variant, features are scaled using `StandardScaler` to check the impact on linear regression.

- **Linear Regression (unscaled data)**:
  - Model: `LinearRegression()`
  - Reported performance:
    - **Training R² ≈ 0.82**
    - **Testing R² ≈ 0.73**
  - Interpretation:
    - The linear model explains a large proportion of the variance in happiness scores.
    - Train vs test R² are reasonably close, suggesting **moderate generalization** without extreme overfitting.

- **Linear Regression (scaled data)**:
  - Model: `LinearRegression()` on scaled features.
  - Reported performance is essentially **the same** as the unscaled version.
  - Interpretation:
    - Scaling the features does **not materially change** the performance of linear regression in this case, which is expected because linear regression is scale-invariant in terms of fit (though scaling can still help numerics or regularized models).

- **Random Forest Regressor**:
  - Model: `RandomForestRegressor(n_estimators=30, max_depth=20)`
  - Reported performance:
    - **Training R² ≈ 0.97** (very high)
    - **Testing R² ≈ 0.69**
  - Interpretation:
    - The random forest fits the training data very closely (almost too well), and test performance is **slightly worse** than linear regression.
    - This indicates some degree of **overfitting** and suggests that a simpler linear model works as well or better for generalization on this dataset.

### Clustering & Elbow Method

The notebook also applies unsupervised learning to group countries into clusters:

- **K-Means clustering**:
  - Model: `KMeans(n_clusters=4)`
  - The algorithm is run on `x` concatenated with `y` (happiness) to cluster countries jointly by predictors and outcome.
  - The resulting cluster assignments `y_means` are examined.

- **Cluster size distribution chart**:
  - A `pd.DataFrame(y_means).value_counts().plot(kind='bar')` chart shows how many observations fall into each cluster.
  - Interpretation:
    - Some clusters contain more countries than others, indicating **dominant happiness profiles** vs rarer ones.

- **Elbow method chart**:
  - For `k` from 1 to 10, K-Means is fitted and the SSE (Sum of Squared Errors) is recorded.
  - A line plot of **SSE vs number of clusters** is generated to form the **elbow curve**.
  - Interpretation:
    - The curve exhibits a clear **elbow around 4 clusters**, supporting the choice `n_clusters=4` as a reasonable balance between compactness and simplicity.

- **Clustered scatter plots**:
  - A grid of scatter plots is created where:
    - X-axis cycles through each key variable (`Logged GDP per capita`, `Social support`, `Healthy life expectancy`, `Freedom to make life choices`).
    - Y-axis is `Ladder score`.
    - Points are colored by **cluster label** (with a `Set1` palette).
  - Interpretation:
    - Clusters correspond to **distinct happiness regimes**: some clusters represent high GDP, high support, high freedom, high happiness countries, while others represent low-scoring profiles.
    - The visualization clearly shows how different combinations of economic, social, and health factors map into different overall happiness clusters.

### Insights Summary

- **Strong drivers of happiness**:
  - GDP per capita, social support, healthy life expectancy, and freedom to make life choices are consistently strong positive predictors of `Ladder score`.
  - Regions with better scores on these indicators have noticeably higher happiness.

- **Weaker drivers**:
  - Generosity is almost uncorrelated with `Ladder score` in a simple linear sense and does not emerge as a strong driver.

- **Model comparison**:
  - Linear regression provides **good and stable performance** (`R²` around 0.73 on test data).
  - Random forest further increases training performance but does **not improve test performance**, suggesting overfitting and reinforcing that the linear relationship is already quite strong.

- **Clustering**:
  - The world’s countries can be **sensibly grouped into about four clusters** based on these variables and happiness scores.
  - These clusters correspond to different overall profiles of prosperity, social support, health, freedom, and happiness.

## Using This Repository

### Prerequisites

- **Python**: 3.8+ recommended  
- Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Make sure you have Jupyter, VS Code, or another notebook environment available.
2. Open `World_Happiness_Report_2021_notebook.ipynb`.
3. Run all cells in order:
   - Data loading and inspection
   - Exploratory charts
   - Correlation and insight cells
   - Model training and evaluation (Linear Regression, Random Forest)
   - Clustering and elbow method visualization
4. Use the charts and printed metrics to explore and confirm the insights summarized above.

### Running the Application (`app.py`)

If you have an `app.py` file at the project root that consumes the processed data and/or trained models (for example, a **Streamlit** or **Flask** app), you can typically run it as follows:

- **Option 1 – Standard Python script (e.g. Flask/FastAPI)**:

```bash
python app.py
```

- **Option 2 – Streamlit app**:

```bash
streamlit run app.py
```

Check the `app.py` file for details on:

- Which host/port it binds to.
- Whether it expects the CSV data or serialized models (e.g. `.pkl` files) to be present in a specific folder.
- Any environment variables or configuration settings it requires.

### Typical Workflow

- **Clone the repo**:

```bash
git clone https://github.com/manishKrMahto/World-Happiness-report-2021.git
cd World-Happiness-report-2021
```

- **Set up a virtual environment (recommended)**:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

- **Explore the analysis**:
  - Open and run `World_Happiness_Report_2021_notebook.ipynb`.

- **Run the app** (if present):
  - Start `app.py` with the appropriate command (see above).

This README is designed to give you both a **high-level overview of the insights** from the notebook (including the main charts and metrics) and **practical instructions** for installing dependencies and using this repository.

# World Happiness Report 2021: Key Insights

## Data Quality and Preparation
The initial dataset was checked for completeness and accuracy:
- No missing values were found, ensuring reliability in analysis.
- The data was cleaned and prepared for deeper exploration.

## Regional Happiness Trends
- The data highlighted that regions like Western Europe and North America scored higher on average in happiness compared to Sub-Saharan Africa and South Asia.
- The top countries with the highest happiness scores included Finland, Denmark, and Switzerland.

## Key Factors Influencing Happiness
- The most significant factors contributing to happiness scores were GDP per capita, social support, and healthy life expectancy.
- Correlation analysis showed a strong positive relationship between GDP per capita and the happiness score.
- Freedom to make life choices and generosity also played notable roles but had smaller impacts compared to GDP and social support.

## Data Visualizations and Observations
- Heatmaps and scatter plots were used to visualize relationships between key metrics, revealing that regions with higher economic strength generally had higher happiness scores.
- Boxplots showed the distribution of happiness scores across different regions, highlighting disparities.

## GDP and Social Support Insights
- Countries with higher GDP per capita generally reported better social support and higher overall happiness scores.
- Western Europe consistently ranked at the top for GDP and social support metrics.

## Life Expectancy and Health
- Healthy life expectancy was a crucial factor; countries with better healthcare systems and higher life expectancy scored better on happiness.
- This trend reinforced the importance of health infrastructure in contributing to the overall happiness of a population.

## Freedom and Generosity Trends
- While freedom and generosity were positively correlated with happiness, the strength of these relationships varied.
- Some countries with lower GDP still showed high happiness scores due to strong social support and perceived freedom.

## Improvement Opportunities
To enhance happiness levels, policymakers could focus on:
- Improving economic stability and growth.
- Enhancing social support networks.
- Promoting healthcare initiatives to increase life expectancy.
- Encouraging freedom of choice and community engagement to strengthen social cohesion.

## Conclusion
The World Happiness Report 2021 provides valuable insights into the factors that contribute to the well-being of populations. Economic prosperity, social support, and health are essential drivers, with opportunities for growth in freedom and generosity metrics.



