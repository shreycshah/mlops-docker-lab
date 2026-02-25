import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == '__main__':
    # Load the Breast Cancer Wisconsin dataset
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Feature names in the dataset
    feature_names = list(cancer.feature_names)

    # Select only 4 features
    selected_features = ["mean radius", "mean texture", "mean smoothness", "mean perimeter"]
    selected_indices = [feature_names.index(f) for f in selected_features]
    X = X[:, selected_indices]

    print(f"Selected features: {selected_features}")
    print(f"Feature indices: {selected_indices}")
    print(f"Dataset shape after selection: {X.shape}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Save scaler mean and scale for use during inference
    np.save('scaler_mean.npy', sc.mean_)
    np.save('scaler_scale.npy', sc.scale_)
    print(f"Scaler mean: {sc.mean_}")
    print(f"Scaler scale: {sc.scale_}")

    # Build a simple TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Save the model
    model.save('breast_cancer_model.keras')
    print("Model was trained and saved as breast_cancer_model.keras")