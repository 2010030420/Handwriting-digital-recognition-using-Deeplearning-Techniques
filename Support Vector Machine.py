from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the MNIST dataset from scikit-learn
mnist = datasets.fetch_openml('mnist_784')

# Prepare the data
X = mnist.data / 255.0  # Scale the data
y = mnist.target.astype(int)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the SVM model
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

# Train the model
svm_model.fit(x_train, y_train)

# Predict on the test set
predicted = svm_model.predict(x_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, predicted)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Plotting accuracy and confusion matrix
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')  # Note: You won't have loss values in SVM
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('SVM Training Loss')  # There's no loss being recorded in SVM
plt.legend()

plt.subplot(1, 2, 2)
# Plot confusion matrix
cm = metrics.confusion_matrix(y_test, predicted)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

plt.tight_layout()
plt.show()
