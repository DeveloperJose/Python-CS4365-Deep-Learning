# Course: CS4365/5354 Topics in Soft Computing: Deep Learning
# Instructor: Dr. Olac Fuentes
# Assignment: Lab 1
# Author: Jose Perez <jperez50@miners.utep.edu>
import numpy as np
import pylab as plt
import scipy.special
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer

def relu(x):
    return x * (x > 0)

program_time = timer()

# Load the files we will be using in this lab
start_time = timer()

x_test = np.loadtxt(fname="dataset/xtest.txt", dtype=np.float64, delimiter=',')
y_test = np.loadtxt(fname="dataset/ytest.txt", dtype=np.uint8, delimiter=',')

h0_w = np.loadtxt(fname="dataset/W0.txt", dtype=np.float64, delimiter=',')
h0_b = np.loadtxt(fname="dataset/B0.txt", dtype=np.float64, delimiter=',')

h1_w = np.loadtxt(fname="dataset/W1.txt", dtype=np.float64, delimiter=',')
h1_b = np.loadtxt(fname="dataset/B1.txt", dtype=np.float64, delimiter=',')

output_w = np.loadtxt(fname="dataset/W2.txt", dtype=np.float64, delimiter=',')
output_b = np.loadtxt(fname="dataset/B2.txt", dtype=np.float64, delimiter=',')

total_time = timer() - start_time
print('Loading of files took %.2fs' % total_time)

# Scale xtest
x_test_scaled = x_test / 255

# Encoding ytest to one-hot encoding
# This uses the fact that the diagonals of the identity matrix is 1
# So 0 = {1,0,0,0,0,0,0,0,0,0} is the same as the first row of the identity
#    1 = {0,1,0,0,0,0,0,0,0,0}
#    2 = {0,0,1,0,0,0,0,0,0,0}
classes = 10
y_test_encoded = np.eye(classes)[y_test]

# Feed-forward computation
relu_func = np.vectorize(relu)
sigmoid_func = scipy.special.expit

h0 = relu_func(np.dot(x_test_scaled, h0_w) + h0_b)
h1 = relu_func(np.dot(h0, h1_w) + h1_b)
output = sigmoid_func(np.dot(h1, output_w) + output_b)

# Accuracy
predictions = np.argmax(output, axis=1)
accuracy = np.count_nonzero(predictions == y_test) / x_test.shape[0]
print("Accuracy %.2f%%" % (accuracy * 100))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Get misclassified digits
indices = (predictions != y_test)
mislabeled_images = x_test[indices]
true_labels = y_test[indices]
predicted_labels = predictions[indices]

# Get the images
# You can use the arrow keys to go through them
mislabeled_images = mislabeled_images.reshape(mislabeled_images.shape[0], 28, 28)

current_im = 0
total_im = mislabeled_images.shape[0]
def toggle_images(event):
    global current_im
    global total_im
    global true_labels
    global predicted_labels
    global mislabeled_images

    if event.key == 'left':
        current_im -= 1
    elif event.key == 'right':
        current_im += 1
    else:
        return

    current_im = current_im % total_im

    # Update plot
    ax.clear()
    print("=====")
    print("True", true_labels[current_im])
    print("Pred", predicted_labels[current_im])
    ax.imshow(mislabeled_images[current_im])
    fig.canvas.draw()

plt.gray()
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', toggle_images)
ax = fig.add_subplot(111)
ax.imshow(mislabeled_images[0])

print("Program took %.2fs" % (timer() - program_time))