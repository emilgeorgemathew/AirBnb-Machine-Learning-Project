# Airbnb Booking Rate Prediction
This repository contains the code and final report for a machine learning project predicting the likelihood of an Airbnb listing achieving a **high booking rate**. Developed for **BUDT 758T: Business Analytics Capstone** at the **Robert H. Smith School of Business, University of Maryland**, this project combines business impact with advanced machine learning to deliver actionable insights for Airbnb stakeholders.

## Table of Contents

* [Business Value and Impact](#business-value-and-impact)
* [Project Overview](#project-overview)
* [Data Description](#data-description)
* [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
* [Modeling and Evaluation](#modeling-and-evaluation)
* [Results](#results)
* [Key Takeaways and Recommendations](#key-takeaways-and-recommendations)
* [Team Contributions](#team-contributions)
* [Repository Contents](#repository-contents)
* [Getting Started](#getting-started)
* [License](#license)

## Business Value and Impact

**This model creates measurable impact for multiple stakeholders:**

* **Airbnb Hosts**: Offers personalized, data-driven suggestions (e.g., instant bookability, pricing ratios, house rules) to improve visibility and boost occupancy by up to 25%, equating to thousands of dollars in annual revenue gains.
* **Airbnb Platform**: Enhances search algorithms, automates listing improvement tips, and drives higher guest satisfaction and platform revenue.
* **Property Managers & Investors**: Acts as a predictive analytics tool to identify high-ROI properties and reduce acquisition risk.

> Our insights bridge the gap between raw listing data and strategic decisions, offering a 15–25% booking rate uplift for optimized listings.

## Project Overview

We framed this problem as a **binary classification task**, predicting whether a listing would receive a **"high booking rate" (YES/NO)**. Working with over 92,000 training samples and 10,000 test listings, we tackled:

* Missing data
* Unstructured text
* Feature explosion
* Model overfitting

The final model is a **stacked ensemble (XGBoost + CatBoost + Logistic Regression meta-learner)** achieving an **AUC of 0.9166** and **Accuracy of 87.30%** on the test set.

![Stacking Ensemble Architecture](images/stacking_architecture.png)

## Data Description

* `airbnb_train_x.csv`: 92,067 training listings
* `airbnb_train_y.csv`: Targets (high\_booking\_rate)
* `airbnb_test_x.csv`: 10,000 listings for prediction
* `data_dictionary.xlsx`: 61 base features covering:

  * **Property details** (type, beds, location)
  * **Host info** (response time, identity verification)
  * **Pricing** (base price, extra charges)
  * **Availability** (calendar windows)
  * **Unstructured text** (summaries, rules, descriptions)

Final engineered dataset: **978 features**, distilled from over 1,200 raw and derived features.

## Data Preprocessing and Feature Engineering

We transformed raw data into rich, predictive features:

### Imputation

* Dropped columns with >85% missingness
* Median/mode imputation or "missing" for text fields

### Feature Engineering

* **Pricing Ratios**: price per bedroom, accommodates, person
* **Log Transformations**: Applied to monetary features
* **Text Analysis**:

  * TF-IDF for amenities, house rules, and verifications
  * Sentiment analysis (TextBlob) on all descriptions
  * Word/character counts
  * Rule parsing: "no smoking", "no parties", etc.
* **Flags & Binaries**: is\_superhost, has\_cleaning\_fee, long\_stay, etc.
* **Categorical Simplification**: grouped rare markets, bed types, property types
* **One-Hot Encoding**: For categorical features

> Each step was validated with data quality checks to ensure no missing values remained.

## Modeling and Evaluation

We applied and compared:

| Model                | AUC        | Accuracy   |
| -------------------- | ---------- | ---------- |
| Logistic Regression  | 0.8292     | 82.00%     |
| Decision Tree        | 0.8443     | 83.02%     |
| KNN                  | 0.7640     | 80.25%     |
| Random Forest        | 0.8814     | 84.38%     |
| **XGBoost**          | 0.9128     | 85.65%     |
| **CatBoost**         | 0.9034     | 86.51%     |
| **Stacked Ensemble** | **0.9166** | **87.30%** |

### Ensemble Details

* **Base Models**: Tuned XGBoost & CatBoost via Optuna
* **Meta-Learner**: Logistic Regression
* **Validation**: 5-fold stratified CV

The ensemble captures non-linear relationships and boosts generalization by combining complementary model strengths.

## Results

* **Test AUC**: 0.9166
* **Test Accuracy**: 87.30%
* **Feature count reduced**: From 1278 to 978
* **Leaderboard Rank**: Top placement in course competition

### Business Insights

* Host response time (within 1 hour) is the strongest driver of high booking rate
* Listings with "real beds", defined cancellation policies, and clear rules perform better
* Competitive pricing and high availability strongly correlate with booking success

## Key Takeaways and Recommendations

**What Worked:**

* Strong pipeline with robust feature engineering
* Aggressive dimensionality reduction
* Extensive hyperparameter optimization with Optuna
* Innovative stacking strategy
* Interpretability through well-curated features

**Challenges:**

* Hyperparameter tuning runtime (many hours)
* Managing 1,278+ features pre-selection
* Integrating multiple ML frameworks (CatBoost, XGBoost, scikit-learn)

**For Future Work:**

* Automate experiments and tuning pipelines
* Use SHAP for deeper interpretability
* Explore neural networks and embedding-based features

## Team Contributions

**Emil George Mathew:** Involved in advanced feature transformation, hyperparameter tuning using Optuna, added TF-IDF features. Model building and evaluation and optimizing AUC score for the competition.

**Chanamallu Venkata Chandrasekhar Vinay:** Contributed to data preprocessing, feature engineering, applied TextBlob sentiment scoring and model evaluation. Worked on implementing ensemble models and optimizing prediction performance.

**Savita Sruti Kuchimanchi:** Supported development of engineered features, sentiment analysis, feature extraction and stacking ensemble architecture. Assisted with training and validation workflows.

**Gurleenkaur Bhatia:** Participated in TF-IDF feature extraction, rule-based feature creation, development of gradient boosting models and tuning of classification models including CatBoost and XGBoost.

## Repository Contents

* `Final Code.ipynb`: Jupyter notebook for full preprocessing, modeling, and prediction
* `Group15_BUDT758T_Final_Report.pdf`: Complete report with graphs, methodology, and results
* `high_booking_rate_group15.csv`: Final predictions on test data

📄 [View Final Report (PDF)](Group15_BUDT758T_Final_Report.pdf)
📘 [Browse the Notebook on nbviewer](https://nbviewer.org/github/yourusername/yourrepo/blob/main/Final%20Code.ipynb)

## Getting Started

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/airbnb-booking-rate-prediction.git
cd airbnb-booking-rate-prediction
pip install -r requirements.txt
```

To run the notebook:

```bash
jupyter notebook "Final Code.ipynb"
```
---

> This project combines business-driven analytics with advanced machine learning to deliver actionable insights for the multi-billion-dollar short-term rental industry.

*Developed for BUDT 758T at the University of Maryland, Robert H. Smith School of Business.*

