import numpy as np
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
import matplotlib
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

simulator = Aer.get_backend('qasm_simulator')

print("1a")
circ = QuantumCircuit(1, 1)
circ.h(0)
circ.measure(range(1), range(1))
circ.draw(output='mpl', filename="1a")

job = execute(circ , simulator , shots=1000)
result = job.result()
counts = result.get_counts(circ)
print(counts)


print("\n1b")
circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure(range(2), range(2))
circ.draw(output='mpl', filename="1b")
job = execute(circ , simulator , shots=1000)
result = job.result()
counts = result.get_counts(circ)
print(counts)

print("\n2")
n_qbits = 6
circ = QuantumCircuit(n_qbits, n_qbits)
for i in range(n_qbits):
    circ.h(i)
circ.measure(range(n_qbits), range(n_qbits))
circ.draw(output='mpl', filename="2")
job = execute(circ , simulator , shots=2**6 * 1000)
result = job.result()
counts = result.get_counts(circ)
print(counts['011111'])
print("expectation value = 1/2^6 * nshots = 1000")

print("\n3 - GHZ")
n_qbits = 5
circ = QuantumCircuit(n_qbits, n_qbits)
circ.h(0)
for i in range(n_qbits-1):
    circ.cx(0,i+1)
circ.measure(range(n_qbits), range(n_qbits))
circ.draw(output='mpl', filename="3_GHZ")
job = execute(circ , simulator , shots=1000)
result = job.result()
counts = result.get_counts(circ)
print(counts)

