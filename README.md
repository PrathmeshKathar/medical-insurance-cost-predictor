# 🏥 Medical Insurance Cost Predictor

An interactive web app built with **Streamlit** that predicts medical insurance charges based on user inputs. It uses a machine learning model trained on the **Medical Cost Personal Dataset** to deliver reliable and insightful cost estimates.

## 🚀 Features

- Predicts medical insurance charges
- User-friendly Streamlit interface
- Real-time results based on user input
- Trained on real-world healthcare cost data

## 📸 Screenshot

<img src="/assets/Screenshot (908).png" alt="App Screenshot" width="700"/>


## 📊 Prediction Model

- **Algorithm**: Random Forest Regressor  
- **Preprocessing**:  
  - `ColumnTransformer` with:
    - `StandardScaler` for numerical features (e.g., age, BMI)
    - `OneHotEncoder` for categorical features (e.g., gender, smoker, region)
- **Pipeline**: End-to-end preprocessing and modeling using Scikit-learn `Pipeline`
- **Hyperparameter Tuning**: `GridSearchCV` with 5-fold cross-validation
- **Model Export**: Serialized using `joblib`  
- **Model File**: `optimized_insurance_model.pkl`

## 🧠 Input Features

- `age`: Age of the individual
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of children covered by health insurance
- `smoker`: Smoking status (yes/no)
- `region`: Residential region in the US (northeast, northwest, southeast, southwest)
