import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from collections import Counter

# Load dataset
data = pd.read_csv('sign_language_data.csv')

# Preprocess data
X = data.drop('label', axis=1).values
y = data['label'].values

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = np_utils.to_categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution
label_counts = Counter(y_train.argmax(axis=1))
print(f"Class distribution: {label_counts}")

# Ensure balanced dataset
if len(set(y_train.argmax(axis=1))) < y_train.shape[1]:
    print("Warning: Some classes are missing in the training data.")

# Build model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')