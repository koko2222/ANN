import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scipy.integrate import odeint

# Parameters
N = 1.40005e9  # Population
Lambda = 0.01 * N  # Birth rate
beta = 0.3  # Transmission rate
d = 0.01  # Natural death rate
delta = 0.05  # Return rate from R to S
a = 0.2  # Progression rate from E to I
r = 0.1  # Recovery rate
m = 0.05  # Disease-induced death rate
eta = 0.1  # Quarantine removal rate
c = [1.0, 1.0, 1.0, 1.0]  # Control cost coefficients
T = 20  # Time horizon
dt = 0.1  # Time step
t = np.arange(0, T + dt, dt)  # Time grid
n_steps = len(t)

# Initial conditions
S0 = 0.99 * N
E0 = 0.01 * N
I0 = 1000
Q0 = 0
R0 = 0
state0 = [S0, E0, I0, Q0, R0]

# Generate baseline data (uncontrolled system)
def seIQR_model(state, t, controls=None):
    S, E, I, Q, R = state
    if controls is None:
        N1, N2, N3, N4 = 0, 0, 0, 0  # No controls
    else:
        N1, N2, N3, N4 = controls
    dSdt = Lambda - beta * S * I - (N1 + d) * S + delta * R
    dEdt = beta * S * I - (N2 + a + d) * E
    dIdt = a * E - (N3 + N4 + r + m + d) * I
    dQdt = N4 * I - (eta + d) * Q
    dRdt = N1 * S + N2 * E + (N3 + m) * I + eta * Q - (delta + d) * R
    return [dSdt, dEdt, dIdt, dQdt, dRdt]

# Solve uncontrolled system
baseline = odeint(seIQR_model, state0, t)
S_base, E_base, I_base, Q_base, R_base = baseline.T

# Neural network model for controls
model = Sequential([
    Dense(64, input_dim=1, activation='relu'),  # Input is time t
    Dense(64, activation='relu'),
    Dense(4, activation='sigmoid')  # Output 4 controls in [0, 1]
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Function to compute system dynamics with neural network controls
def simulate_with_controls(model, t, state0):
    states = np.zeros((len(t), 5))
    states[0] = state0
    controls = np.zeros((len(t), 4))
    
    for i in range(len(t) - 1):
        t_input = np.array([[t[i] / T]])  # Normalize time
        control = model.predict(t_input, verbose=0)[0]
        controls[i] = control
        states[i + 1] = states[i] + np.array(seIQR_model(states[i], t[i], control)) * dt
    
    controls[-1] = model.predict(np.array([[t[-1] / T]]), verbose=0)[0]
    return states, controls

# Custom loss function to approximate J(x, theta)
def compute_loss(model, t, state0):
    states, controls = simulate_with_controls(model, t, state0)
    I = states[:, 2]  # Infected population
    # Approximate integrals using trapezoidal rule
    J_I = np.trapz(I, t)
    J_controls = 0
    for i in range(4):
        J_controls += 0.5 * c[i] * np.trapz(controls[:, i]**2, t)
    return J_I + J_controls

# Training loop
epochs = 50
losses = []
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, t, state0)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.2f}")

# Simulate controlled system
states, controls = simulate_with_controls(model, t, state0)
S, E, I, Q, R = states.T

# Plot state variables
plt.figure(figsize=(10, 6))
plt.plot(t, S_base/N, label='S (Uncontrolled)', linestyle='--')
plt.plot(t, E_base/N, label='E (Uncontrolled)', linestyle='--')
plt.plot(t, I_base/N, label='I (Uncontrolled)', linestyle='--')
plt.plot(t, Q_base/N, label='Q (Uncontrolled)', linestyle='--')
plt.plot(t, R_base/N, label='R (Uncontrolled)', linestyle='--')
plt.plot(t, S/N, label='S (Controlled)')
plt.plot(t, E/N, label='E (Controlled)')
plt.plot(t, I/N, label='I (Controlled)')
plt.plot(t, Q/N, label='Q (Controlled)')
plt.plot(t, R/N, label='R (Controlled)')
plt.xlabel('Time (days)')
plt.ylabel('Population Fraction')
plt.title('SEIQR Model: Controlled vs. Uncontrolled')
plt.legend()
plt.grid()
plt.show()

# Plot controls
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(t, controls[:, i], label=f'Control N{i+1}')
plt.xlabel('Time (days)')
plt.ylabel('Control Value')
plt.title('Neural Network Controls')
plt.legend()
plt.grid()
plt.show()

# Plot loss convergence
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (J)')
plt.title('Loss Convergence Over Epochs')
plt.legend()
plt.grid()
plt.show()