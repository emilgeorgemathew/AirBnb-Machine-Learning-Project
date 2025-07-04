# Airbnb Booking Rate Prediction

This repository contains the code and report for a machine learning project focused on predicting whether an Airbnb listing will have a high booking rate. This project was developed as part of the BUDT 758T class at the Robert H. Smith School of Business, University of Maryland.

## Table of Contents
- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Description](#data-description)
- [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Results](#results)
- [Key Takeaways and Recommendations](#key-takeaways-and-recommendations)
- [Team Contributions](#team-contributions)
- [Files in this Repository](#files-in-this-repository)

## Project Overview

The core objective of this project is to build a robust binary classification model that predicts whether an Airbnb listing will achieve a "high booking rate" (YES/NO). The challenge involved working with a diverse dataset of approximately 92,000 training instances and 10,000 test instances, requiring extensive data preprocessing, feature engineering, and advanced ensemble modeling techniques.

## Business Understanding

A high booking rate prediction model offers significant value to various stakeholders in the short-term rental ecosystem:

**For Airbnb Hosts:** Provides data-informed insights to optimize listing parameters (price, amenities, booking policies) for increased occupancy and revenue. Our analysis suggests optimizations can increase booking rates by 15-25%.

**For Airbnb Corporate:** Enhances platform functionality by improving search algorithms to surface high-potential listings, leading to higher guest satisfaction and increased transaction volume. It also enables automated suggestions to hosts for improving their listings.

**For Property Managers and Investors:** Serves as a portfolio decision support tool, helping to identify high-potential properties for acquisition and prioritize renovations for optimal ROI.

Our model converts complex Airbnb listing data into actionable insights, creating a competitive advantage in a highly competitive market.

## Data Description

The dataset consists of:

- `airbnb_train_x.csv`: Features for 92,067 training listings
- `airbnb_train_y.csv`: Target variables (high_booking_rate and perfect_rating_score)
- `airbnb_test_x.csv`: Features for 10,000 test instances requiring predictions
- `data_dictionary.xlsx`: Descriptions of the 61 features for each listing

The features cover a wide range of attributes, including:

### Property Characteristics
Type, room type, capacity (accommodates, bedrooms, beds, bathrooms), location details (neighborhood, city, state, market, latitude/longitude), amenities, and physical details (square feet).

### Host Information
Basic details (name, location, about, response time), performance metrics (response rate, acceptance rate, listings count), and verification methods.

### Pricing Structure
Base pricing (regular, weekly, monthly), additional fees (security deposit, cleaning fee, extra people charges), and booking requirements (minimum/maximum nights).

### Availability and Policies
30/60/90/365-day availability, cancellation policy, house rules, and experiences offered.

### Text Descriptions
Listing name, summary, space, description, neighborhood overview, transit options, access information, interaction details, and notes.

## Data Preprocessing and Feature Engineering

The raw dataset presented several challenges, including missing values, unstructured text fields, and high-cardinality categorical variables. Our preprocessing pipeline addressed these comprehensively:

### Missing Value Handling
- Dropped columns with excessive missingness (square_feet, license, neighborhood_group)
- Systematic imputation: median for numeric features, mode for categorical features, and "missing" for specific text fields
- Validation: All missing values were successfully handled, resulting in a complete dataset

### Feature Engineering

**Price Ratios:** Derived price_per_bedroom, price_per_accommodates, price_per_person, and price_per_night.

**Log Transformations:** Applied to skewed monetary features (price, cleaning_fee, security_deposit, extra_people, host_listings_count, host_total_listings_count).

**Host Activity:** Calculated host_days_active based on host_since date.

**Listing Age:** Calculated listing_age_days from first_review date.

**Binary Flags:** Created indicators for Host Is Superhost, Instant Bookable, Host Has Profile Pic, Is Location Exact, Host Identity Verified, has_security_deposit, is_weekly_price, is_monthly_price, same_nhood, long_stay, charges_for_extra, has_min_nights, has_cleaning_fee.

### Text Analysis

**TF-IDF Vectorization:** Applied to amenities, host_verifications, and house_rules to extract structured numerical features (e.g., amenity_*, host_verifications_*, house_rules_*).

**Sentiment Analysis:** Calculated polarity scores for space, host_about, summary, description, neighborhood_overview, notes, transit, access, and interaction fields using TextBlob.

**Text Length/Word Count:** Extracted _length and _words features for all descriptive text fields.

**Rule-Based Features:** Parsed common house rules (e.g., "no smoking", "no parties", "no pets") into binary flags.

### Categorical Simplification and Encoding

**Market Grouping:** Consolidated underrepresented cities into an "Other" category.

**Cancellation Policy:** Simplified by consolidating similar policies.

**Property Category:** Grouped similar property types (e.g., 'apartment', 'hotel', 'house', 'unique').

**Bed Category:** Simplified bed_type into 'bed' (for 'Real Bed') and 'other'.

**One-Hot Encoding:** Applied to all relevant categorical features (experiences_offered, host_response_time, city, state, market, property_type, room_type, bed_type, cancellation_policy, charges_for_extra, has_min_nights, host_response, has_cleaning_fee, bed_category, property_category, zipcode_group, host_acceptance, has_security_deposit).

The final dataset comprised 978 features, a significant reduction from the initial high-dimensional space while retaining crucial information for prediction.

## Modeling and Evaluation

Our modeling strategy focused on powerful gradient boosting algorithms and ensemble methods to achieve optimal predictive performance.

### Model Candidates

We experimented with several models:

| Model | AUC | Accuracy |
|-------|-----|----------|
| Logistic Regression | 0.8292 | 82.00% |
| Decision Tree | 0.8443 | 83.02% |
| K-Nearest Neighbors (KNN) | 0.7640 | 80.25% |
| Random Forest | 0.8814 | 84.38% |
| XGBoost Classifier | 0.9128 | 85.65% |
| CatBoost Classifier | 0.9034 | 86.51% |

### Hyperparameter Optimization

We extensively used Optuna for Bayesian hyperparameter optimization across all models. This systematic approach, combined with 5-fold stratified cross-validation, ensured robust parameter selection and prevented overfitting.

### Final Stacking Ensemble (Winner Model)

Our winning model is a stacked ensemble combining the strengths of XGBoost and CatBoost as base learners, with a Logistic Regression meta-learner. This architecture allowed the meta-learner to learn how to optimally combine the predictions of the base models.

**Base Estimators:**
- XGBoostClassifier (with optimized parameters from Optuna)
- CatBoostClassifier (with optimized parameters from Optuna)

**Final Estimator (Meta-Learner):** LogisticRegression

**Cross-Validation:** 5-fold stratified cross-validation

## Results

Our final stacking ensemble model achieved the highest performance in the competition:

- **Final Test AUC:** 0.9166
- **Final Test Accuracy:** 87.30%

This represents a significant improvement over individual models and simpler ensemble approaches. The model demonstrates strong discriminative power and high classification accuracy in identifying Airbnb listings with high booking rates.

The predictions for the test dataset were saved to `high_booking_rate_group15.csv`.

## Key Takeaways and Recommendations

### What Went Well

- **Systematic Model Exploration:** Progressed from simple baselines to complex ensembles, demonstrating clear performance improvements
- **Thorough Preprocessing:** Revisited and refined previous assignments, leading to a robust data pipeline
- **Efficient Hyperparameter Tuning:** Leveraged Optuna for effective exploration of large parameter spaces, optimizing model performance
- **Innovative Ensemble Approach:** Strategically combined XGBoost and CatBoost with a meta-learner, achieving superior results
- **Feature Efficiency:** Achieved better performance with fewer features (978) compared to intermediate models, optimizing for both predictive power and computational efficiency
- **Clear Visualization and Documentation:** Maintained thorough documentation and visualizations of model performance progression

### Main Challenges

- **High Dimensionality:** Managing 1,278 initial features posed computational challenges and increased overfitting risk
- **Prolonged Tuning Cycles:** Hyperparameter optimization for complex models often took many hours, slowing down iteration
- **Balancing Complexity and Performance:** Finding the optimal trade-off between model complexity and performance gains was challenging
- **Feature Engineering Complexity:** Identifying truly meaningful features from raw attributes was time-intensive
- **Technical Integration:** Ensuring compatibility between different machine learning frameworks (XGBoost, CatBoost, scikit-learn) required careful handling

### Lessons Learned and Recommendations for Future Teams

- **Prioritize Rigorous Feature Selection:** Implement aggressive dimensionality reduction early to conserve resources and potentially improve performance
- **Explore Advanced Cross-Validation:** Consider methods like nested cross-validation for more reliable generalization estimates
- **Systematic Data Cleaning:** Employ sophisticated imputation and anomaly detection techniques for enhanced model robustness
- **Automate Experimentation:** Invest in automation for the entire experimentation pipeline from the start for efficiency and reproducibility
- **Utilize Hyperparameter Optimization Tools:** Leverage tools like Optuna for systematic and efficient parameter exploration
- **Consider Computational Efficiency:** Balance performance gains with computational costs; a faster, slightly less accurate model may sometimes be preferable
- **Thoughtful Ensemble Construction:** Combine models based on complementary strengths rather than arbitrary stacking
- **Effective Visualization:** Use clear visualizations to communicate model performance and facilitate informed decision-making

## Team Contributions

**Emil George Mathew:** Involved in advanced feature transformation, hyperparameter tuning using Optuna, added TF-IDF features. Model building and evaluation and optimizing AUC score for the competition.

**Chanamallu Venkata Chandrasekhar Vinay:** Contributed to data preprocessing, feature engineering, applied TextBlob sentiment scoring and model evaluation. Worked on implementing ensemble models and optimizing prediction performance.

**Savita Sruti Kuchimanchi:** Supported development of engineered features, sentiment analysis, feature extraction and stacking ensemble architecture. Assisted with training and validation workflows.

**Gurleenkaur Bhatia:** Participated in TF-IDF feature extraction, rule-based feature creation, development of gradient boosting models and tuning of classification models including CatBoost and XGBoost.

## Files in this Repository

- `Group15_BUDT758T_Final_Report.pdf`: The comprehensive final project report detailing the business understanding, data analysis, methodology, results, and reflections
- `Final Code.ipynb`: A Jupyter Notebook containing the complete Python code for data preprocessing, feature engineering, model development (including hyperparameter optimization), ensemble creation, and prediction generation
- `high_booking_rate_group15.csv`: The final submission file containing the predicted probabilities for the test set

---

*This project was completed as part of BUDT 758T at the Robert H. Smith School of Business, University of Maryland.*
