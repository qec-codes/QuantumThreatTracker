.. quantumthreattracker documentation master file, created by
   sphinx-quickstart on Wed Apr 30 12:11:30 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quantum Threat Tracker documentation
====================================

Welcome to the Quantum Threat Tracker (QTT)!

The Quantum Threat Tracker (QTT) is a Python package designed to provide a
comprehensive overview of quantum threats to cryptographic systems. It helps
researchers, developers, and practitioners understand the potential impact of
quantum computing on current cryptographic standards by estimating the quantum
resources required to break various cryptographic primitives.

The Quantum Threat Tracker (QTT)
---------------------------------

The QTT addresses these challenges by providing a modular framework to:

1.  Estimate the quantum resources required for cryptanalysis.
2.  Provide actionable insights into the vulnerabilities of cryptographic systems.

It achieves this by first determining the quantum resources needed to run a
cryptographically relevant algorithm. Then, by considering quantum computing
hardware roadmaps, it helps estimate when such resources might become available.
Finally, it allows for an assessment of the maximum potential of given quantum
hardware to compromise cryptographic systems, supporting informed decision-making.

For a deeper understanding of the quantum threat landscape and the principles
behind quantum resource estimation, please see our :doc:`background` information.

Overview of QTT Features
------------------------

The QTT is offers three core functionalities:

*   **Quantum Resource Estimation (QRE):** The QTT quantifies the quantum
    resources (e.g., qubit count, runtime) needed to execute
    cryptographic-breaking algorithms like Shor’s for specific protocols and
    key sizes.

*   **Threat Report Generation:** The QTT generates comprehensive threat
    reports predicting when cryptographic systems might become vulnerable. These
    reports are based on hardware roadmaps and algorithmic advancements.

*   **Specification Requirement Estimation:** The QTT allows users to input
    a maximum number of available qubits and determine the hardware
    specifications required to break a given cryptographic protocol.

These features collectively provide a versatile tool for assessing quantum threats.

.. Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.

.. toctree::

   background

.. toctree::
   :maxdepth: 2
   :caption: Sub Modules

   algorithms/index
   lifespan_estimator
   optimizer
   spec_req_estimator