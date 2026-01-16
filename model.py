import pandas as pd
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Load and Preprocess
df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].dropna()

# Convert categorical text to numbers
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

X = df.drop('survived', axis=1)
y = df['survived']

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Train Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Save model AND scaler (We need the scaler to process new inputs in app.py)
# We name it model.h5 as requested
with open('model.h5', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("Success! 'model.h5' for Titanic has been created.")