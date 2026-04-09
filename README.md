# Titanic Survival Prediction using a Bayesian Network

## Overview

This project builds a **Bayesian Network model** to predict passenger
survival on the Titanic using the Kaggle dataset *Titanic: Machine
Learning from Disaster*.

Instead of using common models like logistic regression or XGBoost, this
project explores a **probabilistic graphical model approach** using the
`bnlearn` library to:

-   Learn the network structure from data\
-   Estimate conditional probabilities\
-   Perform probabilistic inference\
-   Compare alternative network structures\
-   Generate survival predictions

------------------------------------------------------------------------

## Dataset

The dataset includes features such as:

-   `Pclass` (Passenger class)
-   `Sex`
-   `SibSp` (Number of siblings/spouses aboard)
-   `Parch` (Number of parents/children aboard)
-   `Embarked` (Port of embarkation)
-   `Survived` (Target variable)

The following columns were removed due to high uniqueness or continuous
scaling:

-   `Name`
-   `Age`
-   `Cabin`
-   `Ticket`
-   `Fare`

------------------------------------------------------------------------

## Methodology

### 1. Data Preprocessing

-   Dropped high-cardinality features
-   Converted categorical variables using `bnlearn.df2onehot`
-   Split data into training and validation sets

### 2. Structure Learning

-   Used Hill Climbing (`methodtype='hc'`)
-   Set `Survived` as the root node
-   Built two models:
    -   **Model 1:** Standard hill climbing
    -   **Model 2:** Hill climbing with `SibSp` blacklisted

### 3. Parameter Learning

-   Learned conditional probability distributions (CPDs) from the
    training data

### 4. Inference

Used Bayesian inference to compute survival probabilities under
different scenarios, such as:

-   Female passenger in 1st class
-   Male passenger in 3rd class

### 5. Prediction

-   Generated predictions for the test dataset
-   Created a Kaggle-style submission file

------------------------------------------------------------------------

## Results

-   Validation Accuracy: \~0.8156
-   Key direct dependencies for `Survived` included:
    -   `Sex`
    -   `Pclass`

Inference results aligned with historical evacuation patterns: - Higher
survival probability for women - Higher survival probability for
first-class passengers

------------------------------------------------------------------------

## Technologies Used

-   Python
-   pandas
-   scikit-learn
-   bnlearn
-   matplotlib
-   seaborn

------------------------------------------------------------------------

## How to Run

1.  Install dependencies:

    ``` bash
    pip install bnlearn pandas scikit-learn matplotlib seaborn
    ```

2.  Place the Titanic train/test CSV files in your working directory.

3.  Run the notebook.

------------------------------------------------------------------------

## Author

Kyuho Park\
Student
