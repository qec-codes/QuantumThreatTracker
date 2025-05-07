Background Concepts
===================

This section provides background on key concepts relevant to the Quantum Threat Tracker.

Understanding the Quantum Threat
--------------------------------

Modern digital security heavily relies on public key cryptography (e.g., RSA,
Diffie-Hellman, elliptic-curve cryptography). These systems are secure because
the underlying mathematical problems (like integer factorization and discrete
logarithms) are computationally intractable for classical computers.

However, the advent of quantum computing, particularly Shor's algorithm (Shor, 1997),
presents a significant challenge. A sufficiently powerful quantum computer could
solve these problems efficiently, potentially compromising encrypted data,
digital signatures, and secure communications. Therefore, understanding the
quantum resource requirements for breaking these protocols is crucial.

While Shor's algorithm laid the theoretical groundwork, ongoing research focuses
on refining and optimizing such quantum algorithms (e.g., Chevignard et al., 2024;
May & Schlieper, 2019; Regev, 2023) for practical quantum hardware.
Implementing these algorithms at scale remains challenging due to the current
limitations of quantum hardware, such as error rates and qubit counts.

Despite these hurdles, continuous advancements in quantum hardware and error
correction suggest that public key encryption systems will face vulnerabilities
in the medium to long term. This necessitates tools like QTT to dynamically
assess these evolving threats based on progress in quantum hardware, error
correction, and algorithmic improvements.

Quantum Resource Estimation (QRE)
---------------------------------

While quantum algorithms like Shor's theoretically demonstrate how to break
public-key cryptography, determining the actual size and capabilities of a
quantum computer needed to execute these algorithms is a complex task. This
complexity arises from the differences between idealized algorithmic assumptions
and the realities of practical quantum computers, which are expected to be
error-prone, have limited connectivity between qubits, and non-negligible gate
operation times.

Quantum error correction (QEC) will be essential for running algorithms like
Shor's, but QEC introduces significant overhead in both the number of physical
qubits required and the overall algorithm runtime.

**Quantum Resource Estimation (QRE)** is the field dedicated to quantifying these
requirements. It involves analyzing quantum algorithms to determine the necessary
resources (such as number of qubits, gate counts, circuit depth, and execution time)
to run them on fault-tolerant quantum computers. QRE is a fundamental tool for
understanding the practical implications of quantum algorithms and is central to
the QTT's goal of assessing the timescales for Cryptographically Relevant
Quantum Computing (CRQC). The QTT leverages and builds upon concepts from QRE
to provide its estimations and threat assessments.
