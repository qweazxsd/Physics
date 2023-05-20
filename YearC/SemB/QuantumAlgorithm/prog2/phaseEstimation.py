from typing_extensions import reveal_type
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
import sys 
from typing import List
from qiskit.visualization import plot_histogram


simulator = Aer.get_backend('qasm_simulator')

def QFT(
        qc: QuantumCircuit,
        qr: List[int],
        inverse = False
        ) -> QuantumCircuit:
    """
    This function takes as arguments a QuantumCircuit and a list of 
    ints which represents the qbits to apply the QFT on. 
    To apply the inverse simply set inverse=True.
    """
    
    n = len(qr)  # number of qbits

    # Apply permutation
    for i in range(n//2):
        qc.swap(i, n-1-i)

    for i, v in enumerate(qr):
        qc.h(v)
        for j in range(n-1-i):
            qc.cp( ((-1)**inverse) * np.pi/(2**abs(i-j)) , v, j)
            

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
    for i in range(n//2):
        qc.swap(i+1, n-i)
    qc.barrier()

    for i in range(1, n+1):
        qc.h(i)
        for j in range(i+1,n+1):
            qc.cp( (-1)*np.pi/(2**abs(i-j)) ,i , j)
        qc.barrier()

    qc.measure(range(1, n+1), range(n))

    job = execute(qc, simulator , shots=10000)
    result = job.result()
    counts = result.get_counts(qc)
    plot_histogram(counts, filename="hist")
    
    dict = {}
    for k, v in counts.items():
        dict[f"{int(k, 2)}"] = v
    plot_histogram(dict, filename="hist2")

    dict = {}
    for k, v in counts.items():
        dict[f"{binary_int(k[::-1])}"] = v
    plot_histogram(dict, filename="hist3")
    qc.draw(output="mpl", filename="Estimate_Circuit")
    return

if __name__ == "__main__":
    phi_in = float(sys.argv[1])
    n = int(sys.argv[2])
    
    estimate(phi_in, n)

