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

    return X, y

# Plot input data.
X,y = usual_function(np.sin, 12, 0.01, 500, [0, 3*np.pi])

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
params, data_training, epochs, accuracies = model.train(X, y, epochs=20000)

# Plot data
folder_name = int(time.time())
if not os.path.exists(f'./data/plots/{folder_name}'):
    os.makedirs(f'./data/plots/{folder_name}/')

# Plot during training
for output, epoch in zip(data_training, epochs):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(X, y, color='lightgrey')
    ax.plot(X, output, color='red')
    ax.title.set_text(f'epoch : {epoch}')
    plt.savefig(f'./data/plots/{folder_name}/{folder_name}_{epoch}.png', dpi=150)

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
ax2.scatter(X,y, s=3, color='lightgrey')
ax2.set_title('Données d\'entraînements', size=8)

ax3 = fig.add_subplot(gs[1:, -1])
ax3.scatter(X,y, s=3, color='lightgrey')
ax3.plot(X, data_training[-1], color='red')
ax3.set_title('Résultats', size=8)

fig.suptitle(f'Entraînement n°{folder_name}')

plt.savefig(f'./data/plots/{folder_name}/summary.png', dpi=150)

# plot input data
plt.clf()
plt.scatter(X,y, color='lightgrey')
plt.title("Données d'entrée")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f'./data/plots/{folder_name}/input_data.png')
