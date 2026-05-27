# 🌸 Iris Dataset — EDA + Decision Tree Classifier

> Exploratory Data Analysis and Classification on the Iris dataset
> using Python and Scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![sklearn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)
![EDA](https://img.shields.io/badge/EDA-13%20Steps-blueviolet)

---

## 📌 About the Project

The Iris dataset is one of the most well-known datasets in machine learning history.
Introduced by statistician Ronald Fisher in 1936, it contains 150 flower samples
from 3 species — Setosa, Versicolor, and Virginica.

Each sample has 4 measurements:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The goal of this project is to:
1. Deeply understand the data through Exploratory Data Analysis
2. Build a Decision Tree Classifier to predict the species
3. Evaluate the model and understand which features matter most

---

## 📁 Project Structure

iris-decision-tree/
│
├── iris_decisionTree.ipynb    ← Main Jupyter Notebook
├── README.md                  ← Project Documentation

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Plotting and visualization |
| Seaborn | Statistical data visualization |
| Scikit-learn | Machine learning — model, metrics, split |

---

## 📊 Dataset Overview

| Property | Detail |
|---|---|
| Source | Seaborn built-in dataset |
| Total Samples | 150 |
| Features | 4 numeric |
| Target Classes | 3 — Setosa, Versicolor, Virginica |
| Samples per Class | 50 each — perfectly balanced |
| Missing Values | None |
| Duplicates | 1 removed |

---

## 🔍 EDA — Step by Step

### Step 1 — Import Libraries
Imported NumPy, Pandas, Matplotlib, and Seaborn
for data handling and visualization.

### Step 2 — Load Dataset
Loaded directly using sns.load_dataset("iris").
No external CSV needed.

### Step 3 — Basic Inspection
Checked shape, column names, data types using df.info() and df.head().
Result: 150 rows, 5 columns, no null values.

### Step 4 — Class Distribution
Used value_counts() to check samples per species.
Result: Exactly 50 samples per class — perfectly balanced.
Balanced data means no class bias during model training.

### Step 5 — Data Quality Check
Checked for missing values → 0 found.
Checked for duplicate rows → 1 found and removed.
Final dataset: 149 clean rows.

### Step 6 — Descriptive Statistics
Used describe() for overall stats.
Used groupby("species").mean() and .std() for per-class stats.

Key finding:
- Setosa petal length mean   = 1.46 cm
- Virginica petal length mean = 5.55 cm
- This 4 cm gap already tells us petal length is a strong
  classifier — before any model is built.

### Step 7 — Boxplots
Plotted each feature against species using boxplots.

What boxplots showed:
- Petal features have very little overlap between species
- Sepal width boxes heavily overlap for all 3 species
- A few mild outliers in sepal width for setosa

### Step 8 — Pairplot
Plotted all feature combinations colored by species.
This is the most comprehensive single visualization.

What pairplot showed:
- Setosa forms a completely separate cluster in all petal-related plots
- Versicolor and Virginica are close but separable
- Diagonal KDE plots confirm petal features have well-separated distributions

### Step 9 — Scatter Plot (Best Separating Features)
Plotted petal_length vs petal_width — the best pair.

What scatter showed:
- Setosa sits in a tight cluster at the bottom left
- Versicolor is in the middle
- Virginica is at the top right
- Only a very small overlap between Versicolor and Virginica

### Step 10 — Correlation Heatmap
Computed correlation between all numeric features.

Key findings:
- petal_length and petal_width: 0.96 correlation — they carry almost the same information
- sepal_width: near zero or negative correlation with all other features
- sepal_width is the least useful feature for classification

### Step 11 — Violin Plots
Combined boxplot + KDE in a single visualization.

What violin plots showed:
- Setosa petal violins are extremely narrow — very consistent, low variation
- Virginica petal violins are wider — more variation in size
- Sepal width violins overlap heavily for all 3 species
- Petal violins clearly show 3 separate shapes at different positions

### Step 12 — Bar Plots
Showed mean values per species for all 4 features.

Key findings:
- Virginica has the highest average petal length and petal width
- Setosa has the smallest petals but the widest sepals

### Step 13 — Histplots with KDE
Plotted frequency distributions with smooth KDE curves.

What histplots showed:
- Petal length and petal width show 3 clearly separated peaks — one per species
- Sepal length and sepal width show heavily overlapping peaks
- Confirms petal features are far better separators than sepal features

---

## 🌳 Decision Tree — Model Pipeline

### Step 14 — Train-Test Split
- 80% training — 119 samples
- 20% testing  — 30 samples
- stratify=y ensures equal class representation in both sets

### Step 15 — Train Decision Tree
- Algorithm : DecisionTreeClassifier
- max_depth : 3 (prevents overfitting)
- criterion : Gini impurity
- random_state : 42

### Step 16 — Predictions
Predicted species labels on the unseen test set.

---

## 📈 Model Results

### Accuracy Score
Test Accuracy → 100%

### Classification Report

| Class      | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Setosa     | 1.00 | 1.00 | 1.00 | 10 |
| Versicolor | 1.00 | 1.00 | 1.00 | 10 |
| Virginica  | 1.00 | 1.00 | 1.00 | 10 |

### Confusion Matrix
All 30 test samples correctly classified.
Zero misclassifications across all 3 species.

---

## 🌲 Decision Tree — How It Decides

Level 1 — Root Split:
→ petal_length <= 2.45?
   YES → Setosa (40 samples, gini = 0.0, perfect separation)
   NO  → move to level 2

Level 2 — Second Split:
→ petal_width <= 1.65?
   YES → likely Versicolor
   NO  → likely Virginica

Level 3 — Final refinement using petal_length thresholds

One single question at the root node perfectly separates
all 40 Setosa samples with zero errors.

---

## 🏆 Feature Importance

| Feature      | Importance Score |
|---|---|
| petal_length | ~0.58 |
| petal_width  | ~0.42 |
| sepal_length | ~0.00 |
| sepal_width  | ~0.00 |

Petal features account for 100% of the model's decisions.
Sepal features contributed nothing to the final tree.
This perfectly matches what EDA showed visually.

---

## 💡 Key Takeaways

1. EDA before modeling is not optional — it is the job.
   The data was already showing the answer through
   visualizations before any model was built.

2. petal_length and petal_width are the two features
   that drive classification in this dataset.

3. sepal_width is the weakest feature — most overlap,
   near-zero importance score.

4. A simple Decision Tree with depth=3 achieves 100%
   accuracy because the natural boundaries in petal
   features are very clean.

5. Good features matter more than complex models.
   The algorithm is just the last step.

---

## 🚀 How to Run

# Clone the repository
git clone https://github.com/raviprakash720/iris-decision-tree

# Move into the folder
cd iris-decision-tree

# Install required libraries
pip install numpy pandas matplotlib seaborn scikit-learn

# Open the notebook
jupyter notebook iris_decisionTree.ipynb

---

## 📬 Connect

If you found this project helpful or have any feedback,
feel free to connect on LinkedIn or raise an issue on GitHub.

---
