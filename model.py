import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. Load data - use ONLY the 10 "mean" features (first 10 columns)
#    so the model matches the 10 inputs collected in app.py
data = load_breast_cancer()
X = data.data[:, :10]      # first 10 columns = the "mean" measurements
y = data.target            # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Scale data (scaler now expects 10 features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Train SVM (probability=True lets us output a real confidence score)
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Save model and scaler
with open('model.h5', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

acc = model.score(scaler.transform(X_test), y_test)
print(f"Success! 10-feature model.h5 created. Test accuracy: {acc:.3f}")
