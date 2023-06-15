from typing_extensions import reveal_type
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
import sys 
from typing import List
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


simulator = Aer.get_backend('qasm_simulator')

def binary_frac_float(s:str) -> float:
    ns = len(s)
    num = 0
    for i in range(ns):
        num += int(s[i])*2**(-(i+1))
    return num

def estimate(phi, n: int):

    qc = QuantumCircuit(n+1, n)  # initializing the circuit

    for i in range(n):
        qc.h(i+1)

    for i in reversed(range(n)):
       for j in range(2**(n-i-1)):
           qc.crz(-4 * np.pi * phi, n-i, 0)
    
    qc.barrier()
    for i in range(n//2):
        qc.swap(i+1, n-i)
    qc.barrier()

    for i in range(1, n+1):
        qc.h(i)
        for j in range(i+1,n+1):
            qc.cp( (-1)*np.pi/(2**abs(i-j)) ,i , j)
        qc.barrier()

    qc.measure(range(1, n+1), range(n))

    job = execute(qc, simulator , shots=2048)
    result = job.result()
    counts = result.get_counts(qc)
    plot_histogram(counts, filename="hist")
    
    dict = {}
    max_counts = 0
    max_bitstring = ""
    max_frac = 0
    for k, v in counts.items():
        x = binary_frac_float(k)
        if v > max_counts:
            max_counts = v
            max_bitstring = k
            max_frac = x
        dict[f"{x}"] = v
    plot_histogram(dict, filename="hist2")

    qc.draw(output="mpl", filename="Estimation_Circuit")
    
    #print(f"most probable bitstring: {max_bitstring}={max_frac}\nwith {max_counts} counts")

    #plt.bar(dict.keys(), dict.values(), label=counts.keys())
    print("phi = ", max_frac)
    plt.show()
    return

if __name__ == "__main__":
    phi_in = float(sys.argv[1])
    n = int(sys.argv[2])
    
    estimate(phi_in, n)

