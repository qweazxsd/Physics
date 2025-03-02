import numpy as np
import matplotlib.pyplot as plt

def free_energy(m, q, T=0.36, J=1):
    energy = -J/2 * ( m**2 + ((1-m)**2)/(q-1) )
    # To avoid issues with log(0), add a small epsilon.
    eps = 1e-12
    entropy = T * ( m * np.log(m + eps) + (1-m) * np.log((1-m)/(q-1) + eps) )
    return energy + entropy

m_vals = np.linspace(0.01, 0.99, 300)
q_values = [2]
T=np.linspace(0.4, 0.6, 6)

plt.figure(figsize=(8,6))
for q in q_values:
    for t in T:
        f_vals = free_energy(m_vals, q, T=t)
        plt.plot(m_vals, f_vals, label=f'q={q}, T={t}')

plt.xlabel('m (fraction of spins in state 1)')
plt.ylabel('Free Energy per spin, f(m)')
plt.legend()
plt.grid(True)
plt.show()

