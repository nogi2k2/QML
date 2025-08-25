# QML Research - Sequential Modelling

This repository documents research conducted on Quantum Recurrent Neural Networks (QRNNs) and related architectures for quantum machine learning applications in sequence modeling tasks, specifically sentiment analysis. The core objective is to explore the feasibility and performance of quantum models relative to recurrent sequence modeling setups, using standardized datasets, controlled training configurations, and trials at building a pure Quantum RNN.

---

## Experimental Overview

Our experimentation has progressed in multiple phases, each focusing on different aspects of architecture design, optimization, and stability:

1. **Initial QRNN Prototypes**  
   Early experiments implemented quantum circuits with simple angle and amplitude encodings. These validated the training pipeline but plateaued at ~50% accuracy, exposing architectural and optimizer limitations.

2. **Critical Optimizer Bug Fix**  
   A breakthrough came from resolving parameter tracking issues:
   - Parameters defined with `numpy` were not updating correctly across QNodes.  
   - Switching to `qml.numpy` with flattened parameter arrays fixed this.  
   - Enabled training progress beyond the 50% ceiling.

3. **Multi-Layer QRNN Architectures**  
   With the optimizer fixed, we explored deeper models. Variants included stacked QRNN layers with intermediate entanglement, enhanced final interaction layers, and minimal entanglement strategies.  
   - Best result: **3-QRNN Minimal** achieved **61% test accuracy** with concentrated final entanglement.  
   - However, manual inference revealed inconsistency (predictions flipped across runs and small input changes).

4. **QRNN (Simulated RNN Block) Variants**  
   In this phase, we attempted to simulate classical RNN dynamics by immediately entangling input qubits with hidden qubits at every timestep, allowing hidden wires to carry combined information.  
   - **QRNN Layer Designs**:  
     - *Simple Rotational Block*: single `qml.Rot`.  
     - *Expressive Block*: RX + RY + RZ rotations → entanglement (CNOT ring or CRY) → RX + RY + RZ rotations.  
   - Experiments spanned 2- and 3-layer variants with consolidation of hidden qubits into the last two readout qubits.  
   - **Results**: None of the models performed consistently well. While QRNN_Sim_v6 peaked at 60% accuracy, repeated tests fluctuated between 54–60%, and increasing shots (>1000) degraded performance further.  
   - **Conclusion**: Despite architectural refinements, the models lacked stability and reliable generalization, underscoring the need for hybridization and error mitigation techniques.

---

## Quantum Model Results 

|       Model        |             Architecture Description                               | Test Accuracy | Train Accuracy |                 Observations                       |
|--------------------|--------------------------------------------------------------------|---------------|----------------|----------------------------------------------------|
| Quantum RNN v1     | Simple Angle Encoding (1 QNode)                                    |     46.5%     |     50.0%      | Stagnated early                                    |
| Quantum RNN v2     | Simple Angle Encoding + 2 Repetitions                              |     46.5%     |     50.0%      | No improvement from repetition                     |
| Quantum RNN v3a    | Dense Angle Encoding                                               |     53.5%     |     51.6%      | Moderate gain                                      | 
| Quantum RNN v3b    | Dense Angle Encoding + 3 Repetitions                               |     53.5%     |     53.5%      | Highest among early QRNNs                          |
| Quantum RNN v4     | Amplitude Encoding                                                 |     53.5%     |     52.8%      | Stable, expressive performance                     |
| Quantum RNN v5     | Hybrid (Amp + Dense Angle Encoding)                                |     52.0%     |     52.0%      | Slight regression                                  |
| Quantum LSTM       | Gate-wise QNodes w/ Amplitude + Angle per gate                     |     53.5%     |     52.8%      | Most consistent generalization (early)             |
| QRNN Adapted       | Dense Angle Encoding, 8D AE input, 2 outputs, CE loss              |     50.1%     |     50.1%      | Stable but underfit                                |
| 2-QRNN + Final Rot | Two QRNN layers + intermediate interactions, final qml.Rot layer   |     58.5%     |     ~58%       | First post-bug-fix success                         |
| 3-QRNN + Final Rot | Three QRNN layers + 2 intermediate interactions + final Rot        |     ~52%      |     ~52%       | Overparameterized, unstable                        |
| 3-QRNN Enhanced    | Three QRNN layers + CNOT ring + RX/RY/RZ final rotations           |     ~55%      |     ~55%       | Better expressivity, training unstable             |
| **3-QRNN Minimal** | 3 QRNN layers, no intermediate entanglement, strong final CNOT ring|   **61.0%**   |     ~60%       | Best performing, consistent statistically          |
| QRNN_Sim Variants  | Input–hidden entanglement, expressive QRNN layers, AE 4D input     |    54–60%     |     ~55–58%    | Unstable, accuracy fluctuated, poor generalization |

---

### Key Observations

- **Optimizer Fix:** Essential for breaking past the 50% plateau.  
- **Minimal Interactions:** Simpler interaction schemes often outperformed complex ones.  
- **QRNN Simulation Findings:** Input–hidden entanglement design failed to deliver stable improvements, highlighting quantum noise sensitivity and shot dependence.  
- **Next Steps:** Focusing on a Hybrid Quantum–Classical model, error mitigation, and stabilization of inference results.  

---

## Experimentation Reports

For deeper insights into the model architectures, training dynamics and analysis:

- [Quantum Phase 1 Report](./Reports/Quantum/Quantum_Report_1.pdf)  
- [Quantum Phase 2 Report](./Reports/Quantum/Quantum_Report_2.pdf)  
- [Quantum Phase 3 Report](./Reports/Quantum/Quantum_Report_3.pdf)  
- [Quantum Phase 4 Report](./Reports/Quantum/Quantum_Report_4.pdf)  

---

## Dataset

> **Note:** This repo contains a large file (`Data.zip`) uploaded via Git LFS.  
> To clone everything follow either:  
> - Run `clone.bat` (requires Git Bash installed)  
> - Manually run `clone.sh` using Bash:  
>   ```bash
>   bash clone.sh
>   ```
