import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import joblib

# Load your training data (replace with your actual data loading logic)
data_path = "Gretel.csv"  # Replace with your data path
x = pd.read_csv(data_path)
df = pd.DataFrame(x)

# Split data into training and testing sets
train, test = df[:160], df[160:]

# Create the PMDARIMA model
m = pm.auto_arima(train, error_action="ignore", seasonal=True, m=12, D=1)  # Enable trace for model diagnostics
x = np.arange(df.shape[0])

plt.plot(x, m.predict(n_periods=df.shape[0]), c='green')
plt.plot(df, c='red')
plt.show()

prediction = pd.DataFrame(m.predict(n_periods=df.shape[0]))
# calculate MAE 
maerror = mae(df,prediction)
  
# MAE = 4174.919775932311
print("Mean absolute error : " + str(maerror))
# Calculate MAPE (not sure)
MAPE = mape(df,prediction)*100
print(f"Mean Absolute Error Percentage: {MAPE:.2f}%")

# Save the trained model to a file
joblib.dump(m, 'arima_model.pkl')

