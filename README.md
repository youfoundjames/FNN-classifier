This project is a simple demonstration of a machine learning pipeline using TensorFlow. The goal is to classify iris species using the famous Iris dataset. In the end, we were able to predict the class of random samples with 100% accuracy.

The dataset contains measurements of 150 iris flowers from three different species. Each sample from the dataset contains 4 features corresponding to petal length, petal width, sepal length, and sepal width. (*Sepals* are the green leafy parts of a flower directly underneath its petals.) Our goal is to build a model that can predict the species of an iris flower based on these measurements.

The data is loaded, the categorical target variable (the species of iris) is encoded, and the data is split into a training set and a test set. A feed-forward neural network is built and compiled using TensorFlow's Keras API. The model has three hidden layers with 10 neurons each and an output layer with 3 neurons (one for each class of iris). The model is trained on the training data using the Adam optimizer and the sparse categorical crossentropy loss function.

To run this project, you will need to have TensorFlow installed in your Python environment. You can install TensorFlow using pip:

```python
pip install tensorflow
```

Once TensorFlow is installed, you can run the Jupyter notebook to go through each step of the machine learning pipeline.
