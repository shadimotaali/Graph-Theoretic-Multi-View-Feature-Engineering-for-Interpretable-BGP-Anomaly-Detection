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
---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.10. GPU (CUDA) is recommended for the GNN experiments but not
required for the flat classifiers.

---

## Data

Raw BGP data (not redistributed here): MRT RIB/UPDATE dumps are public from RIPE RIS. Auxiliary inputs: CAIDA AS-relationships and PeeringDB. See Scripts/ for the fetch utilities and the exact collectors/date ranges used.

---

## Reproducing the results
The pipeline runs in phases (see the paper for details):

```bash
# 1. Feature extraction (graph + statistical views)
python Scripts/...                # or the bgp_*_features notebooks

# 2. Redundancy reduction & complementarity (114 -> 41 -> 22)
python Scripts/cca_ajive_analysis.py

# 3. CORAL domain adaptation + diagnostics
python Scripts/coral_phase2_runner.py
python Scripts/phase2_diagnostics.py

# 4. Supervised transfer sweep (flat classifiers + fusion)
python Scripts/phase3_pipeline.py

# 5. GNN experiments (GAT / TopoGPS)
python Scripts/gnn_classifiers.py --edge astopo_8h_gps ...

# 6. Deployment validation + SHAP
python Scripts/deploy_historical_inference.py
python Scripts/shap_explainability.py
```
All experiments use fixed seeds; see each script's CLI (--help) for configuration.




