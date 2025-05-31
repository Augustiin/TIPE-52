import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageOps
import time
import random
import os
import csv

import neural_net as nn


# Display data
data = []
with open('data/dataset/mnist/mnist_train.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        data.append(row)

data_test = []
with open('data/dataset/mnist/mnist_test.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        data_test.append(row)

# Own images
def process_image(image_path, label):
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.crop((500, 500, 2500, 2500))
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    final_img = Image.new('L', (28, 28), color=0)
    final_img.paste(img, (4, 4))

    pixels = np.array(final_img).astype(np.uint8).flatten()

    for i, pixel in enumerate(pixels):
        if pixel < 75:
            pixels[i] = 0

    return label, pixels.tolist()

# Process data
# own data
paths_own = ['./data/dataset/mnist/IMG_8382.jpg', './data/dataset/mnist/IMG_8383.jpg', './data/dataset/mnist/IMG_8384.jpg']
labels_own = [1, 6, 8]

X_own = []
y_own = []

for path, label in zip(paths_own, labels_own):
    label_own, pixels_own = process_image(path, label)
    X_own.append(pixels_own)
    y_own.append(label_own)

X_own = np.array(X_own, dtype=np.int64)
y_own = np.array(y_own, dtype=np.int64)

# Training data
random.shuffle(data)
data = np.array(data)

X = []
y = []

for row in data:
    X.append([int(value) for value in row][1:])
    y.append(row[0])

X = np.array(X, dtype=np.int64)
y = np.array(y, dtype=np.int64)

# Test data
random.shuffle(data_test)
data_test = np.array(data_test)

X_test = []
y_test = []

for row in data_test:
    X_test.append([int(value) for value in row][1:])
    y_test.append(row[0])

X_test = np.array(X_test, dtype=np.int64)
y_test = np.array(y_test, dtype=np.int64)

# Instantiate the model
model = nn.Model()
# Add layers
model.add(nn.Dense_Layer(784, 128))
model.add(nn.Activation_ReLU())

model.add(nn.Dense_Layer(128, 128))
model.add(nn.Activation_ReLU())

model.add(nn.Dense_Layer(128, 10))
model.add(nn.Activation_SoftMax())

model.set(
    loss=nn.Loss_CategoricalCrossEntropy(),
    optimizer=nn.Optimizer_SGD(learning_rate=0.01),
    type='classification'
)

# Train the model
params_training, data_training, epochs_training, accuracies_training = model.train(X, y, epochs=100, train=True)

# Test the model
print('Testing the model...')
params_testing, data_testing, epochs_testing, accuracies_testing = model.train(X_test, y_test, epochs=1, train=False)

# Plot function
def plot(images, predictions, name, epoch):
    fig, axs = plt.subplots(5, 5, figsize=(10,10))
    for ax, image, prediction in zip(axs.flat, images, predictions):
        image = np.reshape(image, (28,28))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image, cmap='binary')
        ax.set_title(f'Prédiction : {prediction}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(f'Précision : {accuracy:.3f}', fontsize=20, fontweight='bold')
    plt.savefig(f'./data/plots/{folder_name}/predictions/{name}_{epoch}.png', dpi=150)

def plot_confusion_matrix(y_true, y_pred, accuracy, ax=None, save_path=None):

    # Compute confusion matrix
    n_classes = len(np.unique(np.concatenate((y_true, y_pred))))
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1

    # If no axis is provided, create a new figure and axis
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        own_fig = True

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    title_size = 10
    ax_text_size = 5
    if own_fig:
        cbar = fig.colorbar(im, ax=ax)
        title_size = 20
        ax_text_size = 10
    else:
        # If using subplot, attach colorbar to current figure
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=list(range(n_classes)),
        yticklabels=list(range(n_classes)),
        ylabel='Vrai label',
        xlabel='Label prédit',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j],
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', size=ax_text_size)

    # Add accuracy in the title
    ax.set_title(f'Précision: {accuracy:.3f}', fontsize=title_size, fontweight='bold')

    if own_fig:
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)

def plot_summary(images, params, data, y_true, y_pred_i, y_pred_f, accuracy_i, acuracy_f, epochs, folder_name):

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.3)

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.text(0, 0.7, f'Structure : \n{params[0]} \n{params[1]}')
    ax1.text(0, 0.5, f'Loss : {params[3]}')
    ax1.text(0, 0.3, f'Optimizer : {params[4]} & Learning rate : {params[2]}')
    ax1.text(0, 0.1, f'Itérations : {epochs[-1]}')
    ax1.set_title('Paramètres du réseau de neurones', size=10)

    ax2 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix(y_true, y_pred_i, accuracy_i, ax=ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_confusion_matrix(y_true, y_pred_f, acuracy_f, ax=ax3)

    fig.suptitle(f'Entraînement n°{folder_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'./data/plots/{folder_name}/summary.png', dpi=150)


# Plot data
folder_name = int(time.time())
if not os.path.exists(f'./data/plots/{folder_name}'):
    os.makedirs(f'./data/plots/{folder_name}/')
    os.makedirs(f'./data/plots/{folder_name}/confusion_matrix')
    os.makedirs(f'./data/plots/{folder_name}/predictions')

# Plot 25 random during training
images_training = X
for output, epoch, accuracy in zip(data_training, epochs_training, accuracies_training):
    predictions_training = np.argmax(output, axis=1)
    plot(images_training, predictions_training, 'training', epoch)
    plt.close()

# Plot 25 random after testing
images_testing = X_test
predictions_testing = np.argmax(data_testing[0], axis=1)
plot(images_testing, predictions_testing, 'testing', 1)

# Plot confusion matrix
for output, epoch, accuracy in zip(data_training, epochs_training, accuracies_training):
    predictions_training = np.argmax(output, axis=1)
    plot_confusion_matrix(y, predictions_training, accuracy, save_path=f'./data/plots/{folder_name}/confusion_matrix/training_confusion_matrix_{epoch}.png')
    plt.close()

plot_confusion_matrix(y_test, np.argmax(data_testing[0], axis=1), accuracies_testing[0], save_path=f'./data/plots/{folder_name}/confusion_matrix/testing_confusion_matrix.png')

# Plot summary
plot_summary(images_training, params_training, data_training, y, np.argmax(data_training[0], axis=1), np.argmax(data_training[-1], axis=1) ,accuracies_training[0], accuracies_training[-1], epochs_training, folder_name)

# Plot initial data set
images = X
labels = y
fig, axs = plt.subplots(5, 5, figsize=(8,6))
for ax, image, label in zip(axs.flat, images, labels):
    image = np.reshape(image, (28,28))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image, cmap='binary')
    ax.set_title(f'label : {label}', fontsize=6)

plt.tight_layout()
plt.savefig(f'./data/plots/{folder_name}/input_data_labeled.png', dpi=300)

# Plot result for own data
print('Using the model...')
params_using, data_using, epochs_using, accuracies_using = model.train(X_own, y_own, epochs=1, train=False)

images_using = X_own
predictions_using = np.argmax(data_using[0], axis=1)

fig, axs = plt.subplots(1, 3)
for ax, image, prediction in zip(axs.flat, images_using, predictions_using):
    image = np.reshape(image, (28,28))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image, cmap='binary')
    ax.set_title(f'Prédiction : {prediction}')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'./data/plots/{folder_name}/predictions/using.png', dpi=150)
