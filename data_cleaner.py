<<<<<<< HEAD
import pandas as pd

# Load data
df = pd.read_csv("data/earthquake_alert_balanced_dataset.csv")

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Save cleaned dataset
df.to_csv("data/cleaned_earthquakes.csv", index=False)

print(df.info())
=======
import pandas as pd

# Load data
df = pd.read_csv("data/earthquake_alert_balanced_dataset.csv")

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Save cleaned dataset
df.to_csv("data/cleaned_earthquakes.csv", index=False)

print(df.info())
>>>>>>> 967ae3b812f6d3297faaafee1042c64d7e151330
print(df.head())