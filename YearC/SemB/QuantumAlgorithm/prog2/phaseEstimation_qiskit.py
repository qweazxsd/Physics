from typing_extensions import reveal_type
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
import sys 
from typing import List
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT

simulator = Aer.get_backend('qasm_simulator')

def binary_int(s:str) -> int:
    ns = len(s)
    num = 0
    for i in range(ns):
        num += int(s[i])*2**i
    return num

def estimate(phi, n: int):

    qc = QuantumCircuit(n+1, n)  # initializing the circuit

    for i in range(n):
        qc.h(i+1)

    for i in reversed(range(n)):
       for j in range(2**(n-i-1)):
           qc.crz(4 * np.pi * phi, n-i, 0)
    #for i in range(n):
    #   for j in range(2**(i)):
    #       qc.crz(4 * np.pi * phi, n-i, 0)
    
    
    qc.barrier()

    qc.compose(QFT(n, inverse=True), inplace=True, qubits=list(range(1, n+1)))
    qc.measure(range(1, n+1), range(n))

    job = execute(qc, simulator , shots=10000)
    result = job.result()
    counts = result.get_counts(qc)
    plot_histogram(counts, filename="hist")
    
    dict = {}
    for k, v in counts.items():
        dict[f"{k[::-1]}"] = v
    #plot_histogram(dict, filename="hist")

    print(counts)
    qc.draw(output="mpl", filename="Estimate_Circuit")
    return

if __name__ == "__main__":
    phi_in = float(sys.argv[1])
    n = int(sys.argv[2])
    
    estimate(phi_in, n)

