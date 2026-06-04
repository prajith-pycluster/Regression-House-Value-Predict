# California Housing Price Prediction Pipeline

An end-to-end Machine Learning pipeline implemented in scikit-learn to predict California housing prices. This project features structured data fetching, custom feature engineering transformers, a robust data transformation pipeline, and automated hyperparameter tuning using Grid Search with cross-validation.

---

## 🚀 Key Features

* **Stratified Splitting:** Utilizes `StratifiedShuffleSplit` on income categories to eliminate sampling bias and ensure the training/test sets accurately represent the overall population.
* **Custom Scikit-Learn Transformer:** Features a custom engineering class (`AttrCreation`) to automatically extract valuable ratio features (e.g., rooms per household, population density per household).
* **Parallel Transformation Pipeline:** Implements `ColumnTransformer` to handle parallel data pipelines: numerical features are imputed, engineered, and scaled, while categorical elements are dynamically processed using One-Hot Encoding.
* **Hyperparameter Optimization:** Employs `GridSearchCV` paired with a Random Forest Regressor over multiple parameter combinations evaluated via 5-fold cross-validation.

---

## 📊 Pipeline Architecture

The workflow processes raw data into a model-ready state using the following architecture:

```mermaid
graph TD
    A[Raw Data] --> B[Stratified Split]
    B --> C[Separate Features & Labels]
    
    C --> D[Numerical Features]
    C --> E[Categorical Features]
    
    subgraph Numerical Pipeline
        D --> F[Median Imputer]
        F --> G[Feature Engineering AttrCreation]
        G --> H[Standard Scaler]
    end
    
    subgraph Categorical Pipeline
        E --> I[One-Hot Encoder]
    end
    
    H --> J[ColumnTransformer]
    I --> J
    
    J --> K[Random Forest Grid Search]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px

---

## 🛠️ Prerequisites & Installation

To run this pipeline locally, make sure you have Python 3.10+ installed. 

### Dependencies
Install the required packages using pip:

```bash
pip install numpy pandas scikit-learn
