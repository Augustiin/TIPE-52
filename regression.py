from numpy.polynomial import Polynomial as P
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os

import neural_net as nn

def polynomial(pol) : # pol = [1,3,4] ie 1 + 3X^1 + 4X^2

    p = P(pol)
    X = np.linspace(-1, 1, 250).reshape(250, 1)
    y = (p(X) + np.random.normal(0, 0.08, size=X.shape)).reshape(250, 1)

    return X, y

# function : np.fonction, seed : int, noise_factor : [0,1], npoints : nombre de points, range : x_min et x_max liste avec min et max.
def usual_function(function, seed, noise_factor, npoints, range):
    np.random.seed(seed)
    X = np.linspace(range[0], range[1], npoints).reshape(npoints, 1)
    y = (function(X) + np.random.normal(0, noise_factor, size=X.shape)).reshape(npoints, 1)

    return X,y

# Plot input data.
X,y = usual_function(np.sin, 12, 0.01, 700, [0, 3*np.pi])

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

X_train = X[:500]
y_train = y[:500]

X_test = X[500:]
y_test = y[500:]

# Instantiate the model
model = nn.Model()
# Add layers
model.add(nn.Dense_Layer(1, 128))
model.add(nn.Activation_Tanh())

model.add(nn.Dense_Layer(128, 128))
model.add(nn.Activation_Tanh())

model.add(nn.Dense_Layer(128, 1))
model.add(nn.Activation_Linear())

model.set(
    loss=nn.Loss_MeanSquaredError(),
    optimizer=nn.Optimizer_SGD(learning_rate=0.09),
    type='regression'
)

# Train the model
params, data_training, epochs, accuracies = model.train(X_train, y_train, epochs=10000)

# Test the model
print('Testing the model...')
params_testing, data_testing, epochs_testing, accuracies_testing = model.train(X_test, y_test, epochs=1, train=False)

# Plot data
folder_name = int(time.time())
if not os.path.exists(f'./data/plots/{folder_name}'):
    os.makedirs(f'./data/plots/{folder_name}/')

# Plot during training
for output, epoch in zip(data_training, epochs):
    # Reorder data for plotting
    sorted_indices = np.argsort(X_train.flatten())
    X_train_sorted = X_train.flatten()[sorted_indices]
    y_train_sorted = y_train.flatten()[sorted_indices]
    output_sorted = output.flatten()[sorted_indices]

    # Plot data
    fig, ax = plt.subplots(1, 1)
    ax.scatter(X_train_sorted, y_train_sorted, color='lightgrey')
    ax.plot(X_train_sorted, output_sorted, color='red')
    ax.title.set_text(f'epoch : {epoch}')
    plt.savefig(f'./data/plots/{folder_name}/training_{epoch}.png', dpi=150)

# Reorder data for plotting
sorted_indices = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[sorted_indices]
y_test_sorted = y_test.flatten()[sorted_indices]
data_testing_sorted = data_testing[0].flatten()[sorted_indices]

# Plot during testing
fig, ax = plt.subplots(1, 1)
ax.scatter(X_test_sorted, y_test_sorted, color='lightgrey')
ax.plot(X_test_sorted, data_testing_sorted, color='red')
plt.savefig(f'./data/plots/{folder_name}/testing{epochs_testing[0]}.png', dpi=150)

# Summary plot
fig = plt.figure(figsize=(10, 8))

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
ax2.scatter(X_train,y_train, s=3, color='lightgrey')
ax2.set_title('Données d\'entraînements', size=8)


# Reorder data for plotting
sorted_indices = np.argsort(X_train.flatten())
X_train_sorted = X_train.flatten()[sorted_indices]
y_train_sorted = y_train.flatten()[sorted_indices]
final_output_sorted = data_training[-1].flatten()[sorted_indices]

ax3 = fig.add_subplot(gs[1:, -1])
ax3.scatter(X_train_sorted, y_train_sorted, s=3, color='lightgrey')
ax3.plot(X_train_sorted, final_output_sorted, color='red')
ax3.set_title('Résultats', size=8)

fig.suptitle(f'Entraînement n°{folder_name}')

plt.savefig(f'./data/plots/{folder_name}/summary.png', dpi=150)

# plot input data
plt.clf()
plt.scatter(X_train, y_train, color='lightgrey')
plt.title("Données d'entrée")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f'./data/plots/{folder_name}/input_data.png')
