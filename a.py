import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("Dry_Bean_Dataset.csv", nrows=10000)
df["Class"] = pd.Categorical(df["Class"], categories=["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"])
df["Class"] = df["Class"].cat.codes
df.info()
print(df.isnull().sum())
print(df.describe())
pd.crosstab(df["Class"], df["Area"])
print(df.groupby("Class").mean())

iter_vec = []
accuracy_vec = []
results_df = pd.DataFrame(columns=['sample', 'accuracy', 'kernel', 'nu', 'epsilon'])
for sample_num in range(0, 10):
    bestAccuracy = 0
    bestKernel = ""
    bestNu = 0
    bestEpsilon = 0
    iteration = 1000
    kernelList = ['rbf', 'poly', 'linear', 'sigmoid']
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    def create_model(input_shape):
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=input_shape),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(7, activation='softmax')
        ])
        return model

    model = create_model(X_train[0].shape)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32)

    bestAccuracy = max(history.history['val_accuracy'])
    bestKernel = 'unknown'
    bestNu = 0
    bestEpsilon = 0

    iter_vec = np.append(iter_vec, np.arange(1, iteration+1) + (sample_num-1)*iteration)
    accuracy_vec = np.append(accuracy_vec, [bestAccuracy] * iteration)

    new_row = pd.DataFrame([[sample_num, bestAccuracy, bestKernel, bestNu, bestEpsilon]], 
                           columns=['sample', 'accuracy', 'kernel', 'nu', 'epsilon'])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

results_df.to_csv("results.csv", index=False)

max_sample = np.argmax(results_df['accuracy'])
max_iter_vec = iter_vec[(iter_vec > (max_sample - 1)*iteration) & (iter_vec <= max_sample*iteration)]
max_accuracy_vec = accuracy_vec[(iter_vec > (max_sample - 1)*iteration) & (iter_vec <= max_sample*iteration)]
plt.plot(max_iter_vec, max_accuracy_vec, '-o')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
