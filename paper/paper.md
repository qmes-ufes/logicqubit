---
title: 'LogicQubit: A simple and modular framework for simulating quantum algorithms'
tags:
  - Python
  - quantum computing
  - quantum algorithms
  - simulation
  - tensor networks
authors:
  - name: Cleoner S. Pietralonga
    affiliation: 1
  - name: Wendel S. Paz
    affiliation: 1
affiliations:
  - name: Universidade Federal do Espírito Santo (UFES), Brazil
    index: 1
date: 26 January 2026
bibliography: paper.bib
---

# Summary

`LogicQubit` is a simple and lightweight Python framework for the simulation of quantum logic circuits that combines high-precision numerical routines with explicit symbolic representations of states and operators. It supports a dense state-vector backend on the CPU, an optional GPU-accelerated variant based on `CuPy` (`logicqubit-gpu`), and a newer tensor-network backend that enables more efficient simulations of certain many-qubit systems. This unified numerical symbolic design, together with multiple execution backends, is its main distinguishing feature compared to hardware-oriented platforms such as `Qiskit` [@Qiskit] and `Cirq` [@Cirq], while keeping a small code base and a straightforward API.

While many libraries emphasize device interfacing or transpilation to specific quantum processors, `LogicQubit` focuses on a transparent, algebraically consistent formulation of quantum logic. The framework integrates matrix-based simulation, operator algebra, and quantum state visualization within a unified API, enabling construction and analysis of gates, density matrices, and multi-qubit entanglement, as well as analytic inspection of amplitudes, phases, and commutation relations. This makes `LogicQubit` suitable for algorithm prototyping, verification of circuit identities, and conceptual studies on the structure of quantum information, while remaining lightweight and interoperable with established ecosystems such as `NumPy`/`SymPy` and adjacent simulation toolchains (e.g., `QuTiP` [@QuTiP] and `PennyLane` [@PennyLane]).

# Statement of need

Quantum algorithms constitute the core of quantum information science, providing the operational basis for quantum computation and communication. Their formulation relies on constructing circuits composed of logical gates acting on qubit registers, which represent the fundamental unitary transformations of the system. To model and simulate these processes, a variety of open-source frameworks have been developed over the past decade, such as `Qiskit` [@Qiskit], `Cirq` [@Cirq], and `PennyLane` [@PennyLane]. These libraries are optimized for hardware execution, cloud integration, and algorithm benchmarking, forming the technological backbone of today’s quantum ecosystem.

While such platforms are essential for interfacing with real quantum devices, they often rely on complex transpilation pipelines, backend-specific abstractions, and remote execution layers that can obscure the logical structure of the computation. As a consequence, analytical exploration of quantum algorithms, for example studying unitary evolution, phase accumulation, or amplitude interference at the level of individual matrix elements, can become less transparent. For these tasks, a framework that exposes the underlying algebra more directly is desirable.

`LogicQubit` was developed to address this specific need. Built upon `NumPy` and `SymPy`, it provides a lightweight, modular, and fully transparent environment where users can inspect and manipulate qubit amplitudes, density matrices, and operators with explicit control over both their numerical values and symbolic form. Instead of targeting hardware execution, the framework prioritizes introspection and reproducibility, making it particularly useful for theoretical research, algorithmic prototyping, and verification of quantum circuit identities.

Within the current landscape of quantum frameworks, `LogicQubit` occupies a conceptual niche similar to how `tightbinder` (Uría-Álvarez & Palacios, 2024) positioned itself among electronic-structure codes: it does not compete directly with large SDKs but provides alternative tools and a complementary perspective. Where `Qiskit` and `Cirq` excel in quantum hardware integration, `LogicQubit` focuses on exposing the internal algebra of quantum logic and offering a clear view of how information flows through a circuit, now with multiple backend options (CPU, GPU, and tensor networks) tailored to different scales and structures of computation.

Finally, by enabling direct inspection and visualization of quantum states, `LogicQubit` serves as a bridge between abstract mathematical formalism and computational practice. This connection is consistent with the pedagogical tradition established in foundational works such as Nielsen and Chuang’s *Quantum Computation and Quantum Information* [@NielsenChuang], and it reflects the growing importance of transparent, introspective frameworks for the next generation of quantum algorithm development.

# State of the field

The current open-source ecosystem for quantum computing spans hardware-oriented SDKs (e.g., `Qiskit` [@Qiskit], `Cirq` [@Cirq]) and research-focused simulators and differentiable programming toolchains (e.g., `QuTiP` [@QuTiP], `PennyLane` [@PennyLane]). Many widely used platforms emphasize execution on (or compilation toward) specific devices, which can introduce additional abstraction layers.

`LogicQubit` complements this ecosystem by emphasizing an explicit, algebraically transparent representation of states and operators, while still providing multiple numerical backends (CPU, GPU via `CuPy`, and tensor networks) for practical simulation workloads.

# Software design

`LogicQubit` provides a compact yet expressive environment for constructing and analyzing quantum circuits through both numerical and algebraic representations. Its design emphasizes transparency of the underlying quantum operations, enabling users to inspect every stage of computation with minimal abstraction. The main features of the framework include:

- **Unified numerical symbolic simulation.**  
  The package supports direct numerical evolution of quantum states while also allowing amplitudes and operators to be represented symbolically. This unified approach facilitates hybrid workflows in which users can perform algebraic manipulations, verify gate identities, or analytically trace algorithmic steps before or alongside full numerical simulations.

- **Scalable backends on CPU, GPU, and tensor networks.**  
  By default, `LogicQubit` uses a dense state-vector representation on the CPU. The package built on `CuPy` provides a drop-in GPU backend that accelerates linear-algebra operations for larger circuits. In addition, the latest version introduces an optional tensor-network backend, in which the global state is represented as a network of local tensors with bounded bond dimension. For circuits with modest entanglement, this tensor-network representation reduces both memory usage and computational cost relative to a full $2^n$ state-vector simulation, allowing more qubits to be explored on the same hardware.

- **Visualization of states and operations.**  
  `LogicQubit` can plot the current quantum state, the applied operation, the resulting density matrix, and measurement probability distributions. These visualizations provide an immediate and intuitive view of how superposition, interference, and entanglement evolve throughout the computation.

- **Angle-based representation of quantum states.**  
  Qubit amplitudes can be expressed in terms of angular parameters, which is particularly useful in the analysis of algorithms involving phase manipulation, such as the Quantum Fourier Transform (QFT) and phase estimation. This representation allows phase correlations and interference patterns to be examined in a geometrically meaningful way.

- **Flexible control of operations.**  
  Quantum gates can be applied either directly to instantiated `Qubit` objects or by referencing their indices within a `Circuit`. This flexibility allows users to switch easily between object-oriented and index-based paradigms, accommodating different modeling styles or integration with external simulation pipelines.

A typical workflow with `LogicQubit` begins with the definition of a quantum system, where the user specifies the number of qubits and the set of logical operations to be applied. This setup can be performed either interactively within a Python session or programmatically through configuration scripts. Each qubit object maintains an explicit representation of its statevector, allowing the user to apply unitary transformations, controlled gates, and measurements in a fully transparent way.

Once the circuit is defined, the model is instantiated by creating a `LogicQuBit` object and `Qubit` objects and sequentially appending the desired operations. From there, one can evolve the system step by step, inspect intermediate amplitudes or density matrices, and visualize the resulting state and probability distributions. This process is especially useful for investigating interference patterns, phase estimation, and entanglement dynamics in prototype quantum algorithms.

# Code Availability

`LogicQubit` can be installed from PyPI using `pip install logicqubit` for the CPU/tensor-network version and `pip install logicqubit-gpu` for the CuPy-based GPU-enabled variant. For an updated list of functionalities and examples, we recommend visiting the project’s documentation and code in the public repository at https://github.com/qmes-ufes/logicqubit and https://github.com/qmes-ufes/logicqubit-algorithms, where release notes and usage notebooks are maintained.


