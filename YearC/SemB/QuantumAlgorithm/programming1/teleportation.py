import numpy as np
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
import matplotlib
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

simulator = Aer.get_backend('qasm_simulator')

print("\n4 - Quantum Teleportation")
aliceQubits = QuantumRegister(2, 'a')
aliceCbits_0 = ClassicalRegister(1, 'ac0')
aliceCbits_1 = ClassicalRegister(1, 'ac1')
bobQubits = QuantumRegister(1, 'b')
bobCbits = ClassicalRegister(1, 'bc')

circ = QuantumCircuit(aliceQubits, bobQubits, aliceCbits_0, aliceCbits_1, bobCbits)

theta = np.pi/5
phi = np.pi/2
lamda = 0

circ.u(theta, phi, lamda, aliceQubits[0])

print(f"state: {np.cos(theta/2):.3f}|0> + {np.exp(1j*phi)*np.sin(theta/2):.3f}|1>")

expected_p0 = np.cos(theta/2)**2
expected_p1 = np.abs(np.exp(1j*phi)*np.sin(theta/2))**2


circ.barrier()
circ.h(aliceQubits[1])
circ.cx(aliceQubits[1], bobQubits[0])
circ.barrier()
circ.cx(aliceQubits[0], aliceQubits[1])
circ.h(aliceQubits[0])
circ.barrier()
circ.measure(aliceQubits[0], aliceCbits_0[0])
circ.measure(aliceQubits[1], aliceCbits_1[0])
circ.x(bobQubits[0]).c_if(aliceCbits_1, 1)
circ.z(bobQubits[0]).c_if(aliceCbits_0, 1)
circ.measure(bobQubits, bobCbits)
circ.draw(output='mpl', filename="4_teleportation")

n_shots = 100000
print(f"expected probability for 0: {expected_p0}")
print(f"expected probability for 1: {expected_p1}")

job = execute(circ , simulator , shots=n_shots)
result = job.result()
counts = result.get_counts(circ)

tot0 = 0
tot1 = 0
for key, value in counts.items():
    bob_qbit = int(key[0])
    if bob_qbit:
        tot1 += value
    else:
        tot0 += value
print(f"resulting probability for 0: {tot0/n_shots}")
print(f"resulting probability for 1: {tot1/n_shots}")