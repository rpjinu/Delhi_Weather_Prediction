# Delhi_Weather_Prediction
"LSTM-based weather condition prediction using time series meteorological data with label encoding and class balancing."

<img src="https://github.com/rpjinu/Delhi_Weather_Prediction/blob/main/project_image.png">

# 🌦️ Weather Forecasting using LSTM 🧠⏳

A deep learning project for predicting weather conditions based on historical meteorological time series data using an LSTM model. This project includes preprocessing with Label Encoding, scaling, and class balancing to enhance model performance.

---

## 📌 Project Overview

This project uses an **LSTM (Long Short-Term Memory)** model to classify and forecast weather conditions from structured time series data. The key highlights include:

- ⏰ **Time-based prediction** using historical hourly data.
- 🧠 **LSTM model** trained on numerical and encoded categorical features.
- 🧹 Data preprocessing with **Label Encoding**, **scaling**, and **class weight adjustment**.
- 🗂️ Model and encoder serialization for future use.

---

## 📁 Dataset Description

The dataset contains hourly weather records with features such as:

| Feature          | Description                       |
|------------------|-----------------------------------|
| `dewptm`         | Dew point (°C)                    |
| `fog`, `hail`, `rain`, `snow`, `thunder`, `tornado` | Binary indicators (0/1)        |
| `hum`            | Humidity (%)                      |
| `pressurem`      | Atmospheric pressure (mbar)       |
| `tempm`          | Temperature (°C)                  |
| `vism`           | Visibility (miles)                |
| `wdird`, `wspdm` | Wind direction (°), speed (mph)   |
| `hour`, `dayofweek`, `month`, `year` | Date components          |
| `conds`          | Weather condition (target)        |
| `wdire`          | Wind direction (string)           |

### ✅ Preprocessed columns

- `conds_encoded` → Label-encoded weather condition.
- `wdire_encoded` → Label-encoded wind direction.

---

## 🛠️ Project Workflow

### 1. 📚 Data Preprocessing
- Set `datetimeutc` as index.
- Apply `LabelEncoder` to `conds` and `wdire`.
- Save encoders using `joblib` (`label_encoder_conds.pkl`, `label_encoder_wdire.pkl`).
- Scale numerical features using `MinMaxScaler`.

### 2. 🔄 Prepare Time Series Data for LSTM
- Convert to supervised sequence using sliding windows.
- Split data into train and test sets.
- One-hot encode target (`conds_encoded`) for multiclass classification.

### 3. 🧠 Model Architecture (LSTM)

```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, n_features)),
    Dropout(0.3),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
```

- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metrics: `accuracy`

### 4. ⚖️ Class Imbalance Handling
Used `class_weight` to balance training on imbalanced labels.

### 5. 🧪 Evaluation
- Trained for multiple epochs (e.g., 20–50).
- Monitored accuracy and loss on test set.
- Initial accuracy ~58%, improved through hyperparameter tuning and class weighting.

---

## 📈 Results

- ✅ **Test Accuracy**: ~59% (can be improved further)
- 📉 Loss decreases over time with improved tuning.
- Potential for boosting accuracy with:
  - More historical context (longer sequence window).
  - Embedding layers for categorical features.
  - Bidirectional LSTM or attention mechanism.

---

## 📁 File Structure

```
weather-lstm-predictor/
│
├── data/                        # Raw and processed datasets
├── models/
│   ├── lstm_model.h5            # Trained LSTM model
│   └── label_encoder_*.pkl      # Saved encoders
├── notebooks/                   # Jupyter notebooks for EDA and modeling
├── scripts/
│   └── train_lstm.py            # Script to train model
├── README.md
└── requirements.txt
```

---

## 🚀 Future Enhancements

- 🌐 Deploy the model using Streamlit or Flask.
- 📊 Add more visualization for prediction results.
- 🤖 Integrate real-time weather data via API.
- 📦 Support for multiple models (GRU, Transformer).

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/weather-lstm-predictor.git
cd weather-lstm-predictor
pip install -r requirements.txt
```

---

## 📜 Requirements

- Python 3.7+
- TensorFlow / Keras
- pandas, numpy, scikit-learn
- joblib, matplotlib

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork, open issues, or submit PRs

---

## 📌 License

This project is licensed under the MIT License.
