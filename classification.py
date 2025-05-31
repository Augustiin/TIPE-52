import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from nnfs.datasets import vertical_data, spiral_data
import time
import os

import neural_net as nn

X, y = spiral_data(samples=200, classes=3)

# Split data, for training data and testing data.
def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X, y = unison_shuffle(X,y)

X_train = X[:500]
y_train = y[:500]

X_test = X[500:]
y_test = y[500:]

# Instantiate the model
model = nn.Model()
# Add layers
model.add(nn.Dense_Layer(2, 32))
model.add(nn.Activation_ReLU())

model.add(nn.Dense_Layer(32, 32))
model.add(nn.Activation_ReLU())

model.add(nn.Dense_Layer(32, 3))
model.add(nn.Activation_SoftMax())

model.set(
    loss=nn.Loss_CategoricalCrossEntropy(),
    optimizer=nn.Optimizer_SGD(learning_rate=0.9),
    type='classification'
)

# Train the model
params, data_training, epochs, accuracies = model.train(X_train, y_train, epochs=10000, train=True)

# Test the model
print('Testing the model...')
params_testing, data_testing, epochs_testing, accuracies_testing = model.train(X_test, y_test, epochs=1, train=False)

# Plot data
folder_name = int(time.time())
if not os.path.exists(f'./data/plots/{folder_name}'):
    os.makedirs(f'./data/plots/{folder_name}/')

# Plot during training
for output, epoch in zip(data_training, epochs):
    plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(output, axis=1), s=40, cmap='brg')
    plt.title(f'epoch : {epoch}')
    plt.savefig(f'./data/plots/{folder_name}/{folder_name}_{epoch}.png', dpi=150)
    plt.clf()

# Summary plot
fig = plt.figure()
plt.subplots_adjust(hspace=0.1)
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
ax1.text(0, 0.7, f'Structure : \n{params[0]} \n{params[1]}')
ax1.text(0, 0.5, f'Loss : {params[3]}')
ax1.text(0, 0.3, f'Optimizer : {params[4]} & Learning rate : {params[2]}')
ax1.text(0, 0.1, f'Itérations : {epochs[-1]}')
ax1.set_title('Paramètres du réseau de neurones', size=8)

ax2 = fig.add_subplot(gs[1, :-1])
ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap='brg')
ax2.set_title('Données d\'entraînements', size=8)

ax3 = fig.add_subplot(gs[1:, -1])
ax3.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(data_training[-1], axis=1), s=10, cmap='brg')
ax3.set_title('Résultats', size=8)

fig.suptitle(f'Entraînement n°{folder_name}')
plt.savefig(f'./data/plots/{folder_name}/summary.png', dpi=300)

# plot input data
# without classification
plt.clf()
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.savefig(f'./data/plots/{folder_name}/input_data_not_classified.png')

# with classification
plt.clf()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, cmap='brg')
plt.savefig(f'./data/plots/{folder_name}/input_data_classified.png')

# final plot
plt.clf()
plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(data_training[-1], axis=1), s=40, cmap='brg')
plt.savefig(f'./data/plots/{folder_name}/final_data_classified.png')

# Plot input test data
plt.clf()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40, cmap='brg')
plt.savefig(f'./data/plots/{folder_name}/input_test_data_classified.png')

# Plot testing pred
plt.clf()
plt.scatter(X_test[:, 0], X_test[:, 1], c=np.argmax(data_testing[-1], axis=1), s=40, cmap='brg')
plt.title(f'epoch : {epochs_testing[0]}')
plt.savefig(f'./data/plots/{folder_name}/testing.png', dpi=150)
plt.clf()

# results comparaison for test data
y_pred = np.argmax(data_testing[-1], axis=1)
errors = y_pred != y_test

plt.clf()
# Points avec les couleurs des vraies classes
plt.scatter(X_test[:, 0], X_test[:, 1], color='lightgrey', s=40)

# Cercles rouges autour des erreurs
plt.scatter(X_test[errors, 0], X_test[errors, 1], facecolors='none', edgecolors='red', s=80, linewidths=1.5)
plt.savefig(f'./data/plots/{folder_name}/classification_errors_test.png')
