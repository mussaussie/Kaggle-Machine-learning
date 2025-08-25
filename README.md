# 🏠 Housing Price Prediction - Kaggle Learning Project

A machine learning project that predicts housing prices using Decision Trees and Random Forest algorithms. This project demonstrates fundamental ML concepts including model comparison, hyperparameter tuning, and cross-validation.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

## 📊 Project Overview

This project uses the Kaggle House Prices dataset to predict home sale prices. Started as a learning exercise, it evolved to include best practices in machine learning workflow.

### 🎯 Objectives
- Learn fundamental machine learning concepts
- Compare Decision Tree vs Random Forest performance
- Practice hyperparameter tuning
- Implement proper model validation techniques

## 🛠️ Technologies Used

- **Python 3.7+**
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models**: Decision Tree Regressor, Random Forest Regressor
- **Validation**: Train/validation split, 5-fold cross-validation

## 🔍 Features Used

The model uses 7 key features from the dataset:
- `LotArea` - Lot size in square feet
- `YearBuilt` - Original construction date  
- `1stFlrSF` - First floor square feet
- `2ndFlrSF` - Second floor square feet
- `FullBath` - Number of full bathrooms
- `BedroomAbvGr` - Bedrooms above grade (ground level)
- `TotRmsAbvGrd` - Total rooms above grade

## 🚀 Key Features

### ✅ **Bug Fixes**
- Fixed Random Forest evaluation error in original code
- Proper variable naming and model comparison

### 📈 **Model Development**
- Decision Tree with hyperparameter tuning (max_leaf_nodes optimization)
- Random Forest with default and custom parameters  
- Systematic comparison of model performance

### 🔬 **Analysis & Validation**
- Feature correlation analysis with target variable
- Cross-validation for robust performance assessment
- Feature importance ranking (for Random Forest)
- Visualization of hyperparameter tuning results

## 📊 Results

### Model Performance
| Model | Validation MAE | Cross-Validation MAE | Performance |
|-------|----------------|---------------------|-------------|
| Decision Tree (tuned) | ~$25,000 | ~$27,000 | ✅ Good |
| Random Forest | ~$22,000 | ~$24,000 | ✅ Better |

### Key Insights
- **Random Forest outperformed Decision Tree** by ~$3,000 MAE
- **Best Decision Tree**: 100 max_leaf_nodes (optimized via grid search)
- **Most important features**: Square footage and house age were top predictors
- **Model reliability**: Cross-validation confirmed consistent performance

## 🏃‍♂️ How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Steps
1. **Download data** from Kaggle House Prices competition
2. **Place files** (`train.csv`, `test.csv`) in project directory
3. **Run the script**:
   ```bash
   python housing_prediction.py
   ```
4. **Check results**:
   - View performance metrics and visualizations
   - Find predictions in `submission.csv`

## 📁 Project Files

```
📦 housing-price-prediction/
├── 📄 housing_prediction.py          # Main analysis script  
├── 📄 README.md                      # This documentation
├── 📊 train.csv                      # Training data (from Kaggle)
├── 📊 test.csv                       # Test data (from Kaggle)  
└── 📈 submission.csv                 # Final predictions (generated)
```

## 🎓 Learning Outcomes

This project taught me several important ML concepts:

### Technical Skills
- **Hyperparameter Tuning**: Systematic optimization of model parameters
- **Model Comparison**: Proper evaluation and selection methodology  
- **Cross-Validation**: Robust performance assessment techniques
- **Feature Analysis**: Understanding feature importance and correlations

### Best Practices
- **Code Organization**: Clear structure with documented sections
- **Bug Prevention**: Careful variable naming and validation
- **Visualization**: Professional plots for analysis communication
- **Documentation**: Comprehensive project documentation

## 🔧 Original Issues Fixed

1. **Random Forest Bug**: Fixed evaluation using wrong prediction variable
2. **Limited Analysis**: Added correlation and feature importance analysis
3. **No Visualization**: Added plots for hyperparameter tuning and feature importance
4. **Basic Validation**: Enhanced with cross-validation for robust assessment

## 🚀 Future Improvements

- [ ] Add more sophisticated feature engineering
- [ ] Implement additional algorithms (XGBoost, etc.)
- [ ] Include categorical feature handling
- [ ] Add automated hyperparameter optimization (GridSearchCV)
- [ ] Create interactive dashboard for predictions

## 📚 What Made This Project Better

**Before**: Simple model training with basic comparison  
**After**: Professional ML workflow with proper validation and analysis

The enhanced version includes:
- ✅ Proper model evaluation and comparison
- ✅ Systematic hyperparameter optimization  
- ✅ Cross-validation for reliable performance metrics
- ✅ Feature analysis and visualization
- ✅ Professional code structure and documentation

## 🙏 Acknowledgments

- **Kaggle** for the House Prices dataset and learning platform
- **Scikit-learn** for the excellent ML library
- **Data Science community** for best practices and inspiration

---

### 💡 **Perfect for portfolios!** This project demonstrates fundamental ML skills essential for data science roles.

**Skills Demonstrated**: Data preprocessing, model selection, hyperparameter tuning, cross-validation, feature analysis, and professional code documentation.