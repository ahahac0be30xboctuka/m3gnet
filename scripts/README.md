# Experiment protocol and reproduction

[Study overview](../README.md)

## Environment

The archived runs used macOS arm64 with Python 3.10, TensorFlow 2.13.0, M3GNet 0.2.4, pymatgen 2024.2.20 and ASE 3.22.1. From a shell with Conda available, set up the Apple Silicon environment with the following commands.

```bash
conda create --name m3gnet-tf python=3.10 -y
conda activate m3gnet-tf
conda install -c apple tensorflow-deps -y
python -m pip install tensorflow-macos==2.13.0
python -m pip install --no-deps m3gnet==0.2.4
python -m pip install pymatgen==2024.2.20 ase==3.22.1 sympy==1.12
python -m pip install "numpy<1.24" "typing-extensions<4.6"
```

TensorFlow Metal acceleration is optional. On other platforms, use a compatible TensorFlow 2.13 installation. The repository's stored results are from the macOS setup.

The scripts load the pretrained checkpoint by its bundled name, `MP-2021.2.8-EFS`. Set `M3GNET_CHECKPOINT` if you need to provide a different location for those weights. [Upstream checkpoint](https://github.com/materialsvirtuallab/m3gnet/tree/v0.2.4/pretrained/MP-2021.2.8-EFS)

```bash
export M3GNET_CHECKPOINT=/path/to/MP-2021.2.8-EFS
```

## Run the study

Run these commands from the repository root. Each run replaces the corresponding JSON in `scripts/`. Use a separate checkout if you want to keep the archived and new outputs side by side.

```bash
python scripts/expA_predict.py
python scripts/expB_relax.py
python scripts/expC_classes.py
python scripts/expD_prerelax.py
```

| Experiment | Inputs and comparison | Outputs |
| :--- | :--- | :--- |
| A · Force check | Si, Cu, Ni, LiF and Mo at reference lattice parameters | [Script](expA_predict.py) · [JSON](expA_results.json) |
| B · Molybdenum recovery | BCC Mo starting at 3.3 Å and a 2×2×2 supercell with 0.1 Å positional perturbations | [Script](expB_relax.py) · [JSON](expB_results.json) |
| C · Bonding class | Twelve cubic crystals, four per class, initialized at Materials Project lattice parameters | [Script](expC_classes.py) · [JSON](expC_classes_results.json) |
| D · Strain and relaxation | Five metals at two isotropic lattice offsets, with direct scoring and scoring after variable-cell relaxation | [Script](expD_prerelax.py) · [JSON](expD_prerelax_results.json) |

## Metric definitions

**Experiment C.** The response is the absolute lattice deviation, `100 × abs(a_relaxed − a_MP) / a_MP`. Class-level means are descriptive. The input structures and Materials Project IDs are listed alongside the individual outcomes.

**Experiment D.** For each material, the reference energy is the checkpoint's prediction at the reference lattice parameter. The comparison uses the absolute per-atom energy difference from that reference. The finite improvement ratios divide the direct deviation by the relaxed deviation. The script excludes near-zero denominators from its ratio summary and includes all ten cases in both mean deviations.

**Experiment A.** The archived timing field starts after a first energy call on the same ASE object. Treat it as a diagnostic of that execution path. The energy and maximum force norm fields are the outputs used in the study.
