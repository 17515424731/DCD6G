# Toward Causal-Explainable Model Security for 6G DISAC Systems

This repository provides the implementation of **Dynamic Causal Defense (DCD)**
for causal-explainable model security in 6G DISAC systems.

It includes wireless temporal graph construction, trainable baseline models,
TD-PGD evaluation, F1 reporting, model checkpointing, and t-SNE visualization.

## Requirements

- Python 3.9 or newer
- PyTorch
- PyTorch Geometric
- NumPy
- scikit-learn
- matplotlib

Install the required packages:

```bash
pip install -r requirements.txt
```

Set the Python path on Linux or macOS:

```bash
export PYTHONPATH=$PWD/src
```

For Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
```

## Quick Start

The following commands run the complete pipeline using the included sample
trajectory data.

### 1. Build the wireless temporal graph

```bash
python -m dcd6g.wireless \
  --trace-csv data/sample/vehicle_traces_sample.csv \
  --out outputs/disac_temporal_graph_sample.npz \
  --max-distance-m 250 \
  --target-rate-mbps 700 \
  --seed 123
```

### 2. Train and evaluate the models

```bash
python -m dcd6g.wireless_f1_physical \
  --graph outputs/disac_temporal_graph_sample.npz \
  --out-dir outputs/trained_results \
  --epochs 20
```

This command trains and evaluates:

- Vanilla TGNN
- DG-Mamba adaptation
- DCD

### 3. Generate the t-SNE visualization

```bash
python -m dcd6g.make_dual_tsne_paper \
  --embeddings outputs/trained_results/model_embeddings.npz \
  --out-dir outputs/trained_results
```

## Outputs

The main generated files are:

| File | Description |
| --- | --- |
| `disac_temporal_graph_sample.npz` | Generated wireless temporal graph |
| `checkpoints/*.pt` | Trained model checkpoints |
| `wireless_f1_scores.csv` | F1 evaluation results |
| `model_embeddings.npz` | High-dimensional model embeddings |
| `Evaluation_a.pdf` | F1 comparison figure |
| `Evaluation_b.pdf` | t-SNE visualization |
| `dg_mamba_embeddings_tsne.csv` | DG-Mamba t-SNE coordinates |
| `dcd_consistency_embeddings_tsne.csv` | DCD t-SNE coordinates |

## Data Preparation

The expected vehicle trajectory format is:

```csv
timestamp,vehicle_id,x,y
0,vehicle_001,12.4,5.8
1,vehicle_001,13.1,6.2
```

The experiments are based on the London trajectory dataset. Due to dataset size
and licensing restrictions, the complete dataset is not redistributed in this
repository.

Several sample instances are provided under `data/sample/` to demonstrate the
expected data format and code usage. The full dataset should be downloaded from
its official source and used according to the provider's license and access
terms.

To convert the full London trajectory dataset:

```bash
python -m dcd6g.london_traces \
  --input raw_data/london/LondonTrajectories-main/LondonTrajectoriesDataset.csv \
  --out raw_data/vehicle_traces.csv \
  --max-routes 80 \
  --num-steps 120
```

The repository includes representative figures and aggregated result files.
Raw logs, temporary files, and repeated intermediate outputs are not included.

## Repository Structure

| Path | Description |
| --- | --- |
| `src/dcd6g/` | Main implementation |
| `src/dcd6g/full/` | Optional TDAP-compatible utilities |
| `data/sample/` | Sample input data |
| `configs/` | Configuration files |
| `scripts/` | Setup and compatibility scripts |
| `patches/` | Third-party compatibility patches |
| `tests/` | Unit and pipeline tests |

## Optional TDAP Integration

Compatibility helpers for the third-party TDAP pipeline are included:

```bash
bash scripts/fetch_third_party.sh
bash scripts/apply_tdap_patches.sh
```

## Tests

Linux or macOS:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

## Citation

The paper associated with this repository is titled
**Toward Causal-Explainable Model Security for 6G DISAC Systems**.

If you use this code in publicly available work, please cite the paper according
to the citation requirements of the target journal or conference.

## License

The code in this repository is released under the MIT License.

External datasets and third-party repositories are governed by their own
licenses and terms.