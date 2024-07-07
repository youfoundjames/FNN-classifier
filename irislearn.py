# !pip install tensorflow
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# loading and preprocessing dataset (encoding labels)
iris = datasets.load_iris()
inputs = iris.data
labels = iris.target

le = LabelEncoder()
labels = le.fit_transform(labels)

# splitting dataset into train/test
inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.2, random_state=1234)

# building and compiling the model
model = Sequential([
    Input(shape=(4,)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax'),
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] # using a simple accuracy metric
)

# training the model
m = 20 # the model will go through the entire training dataset m times
k = 5 # the model will update its weights after every k samples

history = model.fit(
    inputs_train,
    labels_train,
    validation_data=(inputs_test, labels_test),
    epochs=m,      
    batch_size=k,  
)

# testing the model; looking at its predictions

loss, accuracy = model.evaluate(inputs_test, labels_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# select 10 random samples from the test dataset
random_samples = np.random.choice(len(inputs_test), size=10, replace=False)

for i in random_samples:
    sample = inputs_test[i]
    sample = sample.reshape(1, -1)
    # ^ tensorflow expects the data to be in batched format, not just a 1d array, so we specify:
    # 1 as in 1 sample per batch
    # -1 is a special value that means “infer the size from the length of the array”
    prediction = model.predict(sample)
    predicted_class = np.argmax(prediction)

    print(f"Predicted class: {predicted_class}")

    actual_class = labels_test[i]

    print(f"Actual class: {actual_class}")
    
    if predicted_class == actual_class:
        print("Woohoo!")