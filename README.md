# Graph-Theoretic-Multi-View-Feature-Engineering-for-Interpretable-BGP-Anomaly-Detection
This repository presents an interpretable AS-level BGP anomaly detection framework. It combines graph and statistical features, evaluates their complementarity (CCA/AJIVE), applies CORAL for domain adaptation, compares classical and graph-neural models, and validates performance on real BGP incidents with SHAP explanations.
---

## Overview

Each 5-minute observation window of BGP activity yields **two complementary representations**:
a **graph view** (topology of the target AS's *k*-hop ego network) and a **statistical view**
(control-plane update dynamics). The central finding is that **cross-domain transfer
difficulty is driven by topology asymmetry, not observation-point distance**, which yields a
**topology-driven model-selection rule**: flat classifiers suit sparse, hierarchical ASes;
message-passing graph networks suit dense peering meshes.

**Study domains (collector–AS pairs):**

| ID | Collector | AS      | Operator              |
|----|-----------|---------|-----------------------|
| D1 | RRC04     | AS12880 | Iran TCI (sparse)     |
| D2 | RRC04     | AS3352  | Telefónica (dense)    |
| D3 | RRC05     | AS12880 | Iran TCI              |
| D4 | RRC05     | AS3352  | Telefónica            |
| —  | RRC18     | AS766   | RedIRIS (third-AS validation) |
