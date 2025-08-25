# QML Research - Sequential Modelling

This repository documents research conducted on Quantum Recurrent Neural Networks (QRNNs) and related architectures for quantum machine learning applications in sequence modeling tasks, specifically sentiment analysis. The core objective is to explore the feasibility and performance of quantum models relative to recurrent sequence modeling setups, using standardized datasets, controlled training configurations, and trials at building a pure Quantum RNN.

---

## Experimental Overview

Our experimentation is currently in progress, and the results provided are subject to updates as we refine the architectures. Work so far has proceeded in multiple phases:

1. **Initial QRNN Prototypes**  
   Early experiments implemented quantum circuits with simple angle and amplitude encodings. These served to validate the training pipeline but plateaued around 50% accuracy, highlighting a limitation in both architecture and optimizer setup

2. **Critical Optimizer Bug Fix**  
   A significant breakthrough came when resolving a parameter tracking issue:
   - Parameters defined with standard `numpy` were not updating correctly across QNodes  
   - Switching to `qml.numpy` with flattened parameter arrays fixed the issue  
   - This unlocked meaningful training progress and enabled performance improvements beyond the 50% ceiling

3. **Multi-Layer QRNN Architectures**  
   After the optimizer fix, experiments explored stacking QRNN layers with different interaction schemes. Variants included intermediate entanglement layers, enhanced final interaction layers, and minimal entanglement strategies  

---

## Quantum Model Results 

|       Model        |             Architecture Description                               | Test Accuracy | Train Accuracy |      Observations                        |
|--------------------|--------------------------------------------------------------------|---------------|----------------|------------------------------------------|
| Quantum RNN v1     | Simple Angle Encoding (1 QNode)                                    |     46.5%     |     50.0%      | Stagnated early                          |
| Quantum RNN v2     | Simple Angle Encoding + 2 Repetitions                              |     46.5%     |     50.0%      | No improvement from repetition           |
| Quantum RNN v3a    | Dense Angle Encoding                                               |     53.5%     |     51.6%      | Moderate gain                            | 
| Quantum RNN v3b    | Dense Angle Encoding + 3 Repetitions                               |     53.5%     |     53.5%      | Highest among early QRNNs                |
| Quantum RNN v4     | Amplitude Encoding                                                 |     53.5%     |     52.8%      | Stable, expressive performance           |
| Quantum RNN v5     | Hybrid (Amp + Dense Angle Encoding)                                |     52.0%     |     52.0%      | Slight regression                        |
| Quantum LSTM       | Gate-wise QNodes w/ Amplitude + Angle per gate                     |     53.5%     |     52.8%      | Most consistent generalization (early)   |
| QRNN Adapted       | Dense Angle Encoding, 8D AE input, 2 outputs, CE loss              |     50.1%     |     50.1%      | Stable but underfit                      |
| 2-QRNN + Final Rot | Two QRNN layers + intermediate interactions, final qml.Rot layer   |     58.5%     |     ~58%       | First post-bug-fix success               |
| 3-QRNN + Final Rot | Three QRNN layers + 2 intermediate interactions + final Rot        |     ~52%      |     ~52%       | Overparameterized, unstable              |
| 3-QRNN Enhanced    | Three QRNN layers + CNOT ring + RX/RY/RZ final rotations           |     ~55%      |     ~55%       | Better expressivity, training unstable   |
| **3-QRNN Minimal** | 3 QRNN layers, no intermediate entanglement, strong final CNOT ring|   **61.0%**   |     ~60%       | Best performing, consistent statistically|

---

### Key Observations:

- **Optimizer Fix:** Correct parameter tracking was critical for breaking past the 50% plateau  
- **Minimal Interactions Win:** Reducing intermediate entanglement and concentrating complexity in the final interaction layer (CNOT ring + composite rotations) yielded the best results 
- **Best Performance:** 3-QRNN Minimal achieved **61.0% test accuracy** with ~60% train accuracy  
- **Performance–Consistency Paradox:** Despite statistical success, manual inference showed instability:
  - Classifications varied across runs  
  - Minor sentence rewording could flip predictions  
  - Suggests reliance on dataset-specific artifacts instead of robust sentiment features  

---

## Experimentation Reports

For deeper insights into the model architectures, training dynamics and analysis:

- [Quantum Phase 1 Report](./Reports/Quantum/Quantum_Report_1.pdf)
- [Quantum Phase 2 Report](./Reports/Quantum/Quantum_Report_2.pdf)
- [Quantum Phase 3 Report](./Reports/Quantum/Quantum_Report_3.pdf)

---

## Dataset

> **Note:** This repo contains a large file (`Data.zip`) uploaded via Git LFS.  
> To clone everything follow either:  
> - Run `clone.bat` (requires Git Bash installed)  
> - Manually run `clone.sh` using Bash:  
>   ```bash
>   bash clone.sh
>   ```

---
