# AI vs Human Text Classification

## Overview

This project analyzes a benchmark dataset containing **AI-generated and
human-written text**. The notebook performs exploratory data analysis,
data-quality checks, statistical analysis, visualization, and evaluation
of an existing benchmark detection score.

The main notebook is:

-   `AI__VS_Human_Text_Classification.ipynb`

## Objectives

The notebook focuses on:

-   Understanding the structure and quality of the dataset
-   Checking missing values and duplicate records
-   Analyzing the distribution of the target label
-   Comparing numerical text features between human and AI-generated
    text
-   Examining differences across AI source models
-   Evaluating the benchmark detection score using ROC-AUC
-   Studying correlations between numerical features
-   Detecting potential outliers
-   Performing raw-text analysis using tokenization, stop-word removal,
    and word clouds

## Dataset

The notebook expects the following CSV file:

``` text
AI_vs_Human_Text_Classification_Benchmark.csv
```

The dataset is loaded from:

``` text
/content/AI_vs_Human_Text_Classification_Benchmark.csv
```

when the notebook is run in Google Colab.

The notebook uses a binary `Label` column:

-   `0` --- Human-written text
-   `1` --- AI-generated text

The notebook also works with text and metadata/features such as:

-   `Raw_Text`
-   `Source_Model`
-   `Model_Size`
-   `Generation_Prompt_Type`
-   `RAG_Context_Provided`
-   `Model_Temperature`
-   `Word_Count`
-   `Lexical_Diversity_Score`
-   `Burstiness_Score`
-   `Coherence_Score`
-   `Formal_Tone_Score`
-   `Detection_Score_Benchmark_Model_A`

## Analysis Workflow

### 1. Setup and Data Loading

The notebook installs and imports the required Python libraries and
loads the benchmark CSV dataset.

### 2. Data Quality Checks

The notebook checks:

-   Missing values
-   Structural missing values in `Model_Temperature`
-   Duplicate rows
-   Human vs AI row counts

### 3. Target Analysis

The target variable is analyzed using:

-   Label counts
-   Target-proportion visualization
-   Cross-tabulations with metadata columns
-   Data-leakage checks

### 4. Univariate Analysis

Numerical features are explored using distributions and
boxplots/countplots to understand their individual behavior.

### 5. Human vs AI Comparison

Numerical features are compared between the two classes using
visualizations such as density plots, count plots, violin/box plots, and
distribution comparisons.

### 6. AI Model-wise Analysis

For AI-generated text, selected metrics are compared across different
source models. The notebook also examines the relationship between model
temperature and selected text characteristics when sufficient data is
available.

### 7. Detection Score Analysis

The notebook evaluates:

``` text
Detection_Score_Benchmark_Model_A
```

and calculates an ROC curve and ROC-AUC score. If the score direction is
inverted, the notebook adjusts the score direction before calculating
the final ROC-AUC.

### 8. Correlation Analysis

The notebook generates:

-   A Pearson correlation matrix
-   Feature-to-target correlation analysis

These analyses help identify relationships among numerical features and
their association with the classification label.

### 9. Outlier Analysis

The notebook uses the **IQR (Interquartile Range)** method to identify
potential outliers in numerical features and summarizes outlier counts
for the overall dataset and each class.

### 10. Raw Text Analysis

The `Raw_Text` column is analyzed using:

-   English stop-word removal
-   Regex-based tokenization
-   Human vs AI token collections
-   Word-frequency analysis
-   Word clouds

Sampling is used in parts of the text analysis to limit processing to a
manageable number of records.

## Technologies and Libraries

The project uses Python and the following libraries:

-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   SciPy
-   Scikit-learn
-   NLTK
-   WordCloud

## Installation

Install the required dependencies with:

``` bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn nltk wordcloud
```

If you are using Jupyter Notebook:

``` bash
pip install notebook
```

## How to Run

### Option 1: Google Colab

1.  Upload the notebook to Google Colab.
2.  Upload `AI_vs_Human_Text_Classification_Benchmark.csv`.
3.  Run the notebook cells from top to bottom.

### Option 2: Local Jupyter Notebook

Clone or download this repository and place the CSV dataset in the
location expected by the notebook, or update the `file_path` variable in
the notebook.

Then start Jupyter:

``` bash
jupyter notebook
```

Open:

``` text
AI__VS_Human_Text_Classification.ipynb
```

and run the cells sequentially.

## Repository Structure

``` text
AI-VS-Human-Text-Classification/
│
├── AI__VS_Human_Text_Classification.ipynb
├── AI_vs_Human_Text_Classification_Benchmark.csv
└── README.md
```

> The CSV dataset is shown in the structure above because the notebook
> depends on it. Add it to the repository only if its licensing and
> distribution terms allow you to do so.

## Notes

-   The notebook contains checks for missing data and structural `NaN`
    values.
-   Some analyses use samples of the dataset to reduce computational
    cost.
-   The notebook is primarily an analysis and evaluation workflow; it
    does not train a new machine-learning classifier in the visible
    analysis cells.
-   The final summary is generated dynamically from the data when the
    notebook is executed.

## Future Improvements

Possible extensions include:

-   Training supervised machine-learning classifiers
-   Adding TF-IDF or transformer-based text features
-   Comparing multiple detection models
-   Performing cross-validation
-   Reporting precision, recall, F1-score, and confusion matrices
-   Testing the system on an independent holdout dataset
-   Building a simple web interface for AI-vs-human text prediction

## License

Add an appropriate license to this repository based on the licensing
terms of the notebook and dataset.
