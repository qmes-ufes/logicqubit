# Overview

LogicQuBit is a lightweight quantum computing toolkit written in pure Python. It provides a high-level API for building quantum circuits, applying quantum gates, performing measurements, and inspecting the resulting quantum state. Unlike many simulation frameworks that focus exclusively on numerical amplitudes, LogicQuBit lets you switch between numerical and symbolic modes, making it possible to reason about circuits algebraically or compute exact matrix representations. Under the hood, the project leverages TensorNetwork backends to run your simulations on numpy, jax, pytorch or tensorflow, and it can optionally use CUDA acceleration when a compatible GPU is available.

LogicQuBit is ideal for educators, researchers and enthusiasts who want to explore quantum algorithms without the overhead of a full-blown SDK. It supports a wide range of single, two and three-qubit gates, built-in measurement operations, density-matrix and purity calculations, and tools for visualising quantum states.

## Features

- **Numerical and symbolic** simulation of quantum circuits and algorithms
- Apply gates either:
  - directly on **`Qubit` objects**, or
  - via the **`LogicQuBit`** circuit object using qubit indices
- **State inspection tools**:
  - print the current state (pretty LaTeX in notebooks)
  - view the **density matrix**
  - inspect **phases/angles** of amplitudes (useful when studying the **Quantum Fourier Transform**)
- **Measurement utilities**:
  - expected-value / probability distribution for one or more qubits
  - single-shot measurement with state collapse
- Optional numeric backend configuration via **TensorNetwork** (and optional CUDA-enabled torch)

---

# Table of Contents
- [Installation](#installation)
- [Startup](#startup)
  * [To instantiate a qubit](#to-instantiate-a-qubit)
  * [To instantiate a qubit register](#to-instantiate-a-qubit-register)
- [Operations](#operations)
  * [Operations on one qubit](#operations-on-one-qubit)
  * [Operations on two qubits](#operations-on-two-qubits)
  * [List of available gates](#list-of-available-gates)
- [Measure](#measure)
  * [Measure the expected value of one or more qubits](#measure-the-expected-value-of-one-or-more-qubits)
  * [Measure one shot on a qubit](#measure-one-shot-on-a-qubit)
- [Plot graphs and print state](#plot-graphs-and-print-state)
  * [Plot expected values](#plot-expected-values)
  * [Plot the density matrix](#plot-the-density-matrix)
  * [Print the current state](#print-the-current-state)
  * [Print the current state as angles](#print-the-current-state-as-angles)
- [Code sample](#code-sample)
- [Other code samples](#other-code-samples)

---

# Installation

CPU-only install:

```bash
pip install logicqubit
````

GPU (PyTorch/CUDA) extras:

```bash
pip install "logicqubit[cuda]"
```

> Notes:
>
> * `logicqubit[cuda]` installs **PyTorch**. For other TensorNetwork backends (e.g. JAX / TensorFlow), install them separately.
> * Plotting uses `matplotlib`.

---

# Startup

Create a circuit with `n_qubits`. By default it runs in **numeric** mode.

```python
from logicqubit.logic import LogicQuBit, Qubit, QubitRegister

logic = LogicQuBit(n_qubits=3)  # numeric by default
```

Enable **symbolic** mode (state coefficients are created symbolically using SymPy):

```python
logic = LogicQuBit(3, symbolic=True)
```

Additional optional keyword arguments let you choose the TensorNetwork backend and toggle CUDA acceleration:

```python
logic = LogicQuBit(
    3,
    tn_backend="numpy",   # or "jax", "pytorch", "tensorflow"
    enable_cuda=False,    # set True to try a GPU-capable backend (if available)
    first_left=True       # qubit 1 is left-most in the tensor product
)
```

> Important: `LogicQuBit(n_qubits, ...)` defines how many qubits exist in the session. After that, you can instantiate `Qubit()` objects that automatically get assigned available IDs.

## To instantiate a qubit

```python
q = Qubit()
```

## To instantiate a qubit register

A `QubitRegister(n)` creates a group of `n` qubits and lets you apply the same single-qubit gate to all of them.

```python
reg = QubitRegister(4)

# Apply H to all qubits in the register
reg.H()
```

> Register indexing is **1-based**:
> `reg[1]` is the first qubit in the register, `reg[2]` the second, etc.

---

# Operations

You can apply gates in two equivalent styles:

1. **Circuit style** (pass qubit indices or `Qubit` objects):

```python
logic.H(1)
logic.CX(1, 2)
```

2. **Object style** (call the gate method on a `Qubit` instance):

```python
q1 = Qubit()
q2 = Qubit()

q1.H()

# For 2-qubit controlled gates, call the method on the TARGET and pass the CONTROL
q2.CX(q1)     # control=q1, target=q2
q2.CU1(q1, 0.5)
```

> Tip: IDs are **1..n_qubits**. You can pass either integers or `Qubit` objects; LogicQuBit internally converts qubits to IDs.

## Operations on one qubit

* `q.Gate(...)` (object style), or
* `logic.Gate(target, ...)` (circuit style)

Example:

```python
logic = LogicQuBit(1)
q = Qubit()

q.H()
q.RZ(1.234)
```

## Operations on two qubits

* `target.Gate(control, ...)` (object style), or
* `logic.Gate(control, target, ...)` (circuit style)

Example:

```python
logic = LogicQuBit(2)
q1, q2 = Qubit(), Qubit()

q2.CX(q1)         # CNOT with q1 as control, q2 as target
logic.CZ(q1, q2)  # same operation style, but via the circuit object
```

*The need for parameters depends on the gate.

## List of available gates

**Single-qubit gates**

* `X`, `Y`, `Z`, `V`, `S`, `T`, `H`
* `RX`, `RY`, `RZ`
* `U`, `U1`, `U2`, `U3`

**Two-qubit gates**

* `CH`
* `CX` (or `CNOT`)
* `CY`, `CZ`
* `CV`, `CS`, `CT`
* `CRX`, `CRY`, `CRZ`
* `CU`, `CU1`, `CU2`, `CU3`
* `SWAP`

**Three-qubit gates**

* `CCX` (or `Toffoli`)
* `Fredkin`

---

# Measure

## Measure the expected value of one or more qubits

`Measure([...])` returns the probability distribution over all bitstrings for the chosen qubits.

```python
probs = logic.Measure([q1, q2, q3])
```

If you want the first qubit in your list to be interpreted as the **most significant bit**, use:

```python
probs = logic.Measure([q1, q2, q3], fisrt_msb=True)
```

## Measure one shot on a qubit

`Measure_One` samples a measurement outcome (optionally multiple shots) and collapses the state:

```python
value = logic.Measure_One(q1)        # returns a list with one sampled bit (e.g. [0] or [1])
value = logic.Measure_One(q1, shots=10)
```

---

# Plot graphs and print state

## Plot expected values

After calling `Measure(...)`, you can plot the measured distribution:

```python
logic.Plot()               # bar chart over measured basis states
logic.Plot(big_endian=True)
```

## Plot the density matrix

```python
logic.PlotDensityMatrix()                 # real part by default
logic.PlotDensityMatrix(imaginary=True)   # imaginary part
```

## Print the current state

In notebooks, `PrintState()` uses LaTeX rendering:

```python
logic.PrintState()
```

For a plain-text output:

```python
logic.PrintState(simple=True)
```

## Print the current state as angles

`getPsiAtAngles()` returns the phase angles of the state vector coefficients:

```python
angles = logic.getPsiAtAngles(degree=True)   # degrees (or degree=False for radians)
print(angles)
```

---

# Code sample

```python
from logicqubit.logic import *
from scipy.linalg import expm
import numpy as np

# Inverse Quantum Fourier Transform
def iqft(qr):
    n = len(qr)
    for i in range(n):
        for j in range(i):
            qr[i].CU1(qr[j], -np.pi / float(2**(i - j)))
        qr[i].H()

gates = Gates(1)

X = gates.X()
Y = gates.Y()

# Hamiltonian and unitary U
H = (Y + X * 2.2).get()
H = np.array(H.tolist(), dtype=complex)

tau = np.pi / (2 * np.sqrt(2))

# Time-evolution unitary U = exp(i * tau * H)
U = expm(1j * tau * H).tolist()   # LogicQuBit expects a list-of-lists

# Registers and circuit
logicQuBit = LogicQuBit(5)

# First 4 qubits = phase register, last qubit = eigenstate register
x1, x2, x3, x4, y = [Qubit() for _ in range(5)]

# Put the phase register in uniform superposition
for x in (x1, x2, x3, x4):
    x.H()

# Controlled powers of U
for k, c in enumerate([x4, x3, x2, x1]):
    for _ in range(2**k):
        y.CU(c, U)

# Apply inverse QFT on the phase register
iqft([x1, x2, x3, x4])

# Measurement and post-processing
probs = logicQuBit.Measure([x1, x2, x3, x4])
logicQuBit.Plot()

n = 4  # number of phase qubits
print("Result:")
for idx, p in enumerate(probs):
    if p > 1e-1:
        phi = idx / 2**n
        phi_sgn = phi - 1 if phi > 0.5 else phi
        E = 2 * np.pi * phi_sgn / tau
        print(f"|{idx:04b}>  phi={phi_sgn:+.3f}  E={E:+.6f}")

Result:
|0111>  phi=+0.438  E=+2.474874
|1001>  phi=-0.438  E=-2.474874
```

![](https://github.com/clnrp/logicqubit/blob/master/images/1769913080.png)

> This example uses `scipy.linalg.expm` (SciPy is not a strict dependency of LogicQuBit, so install it if needed).

---

# Other code samples

* [https://github.com/clnrp/logicqubit-algorithms](https://github.com/clnrp/logicqubit-algorithms)
* See also the `examples/` folder in this repository for:

  * Grover (2 qubits), phase kickback, phase estimation variants, symbolic examples, and truth-table/oracle demos.

