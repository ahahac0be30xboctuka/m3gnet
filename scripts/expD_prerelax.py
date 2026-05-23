import os
import warnings
import json
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from m3gnet.models import M3GNet, Potential, M3GNetCalculator, Relaxer
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get("M3GNET_CHECKPOINT", "MP-2021.2.8-EFS")
print("Loading M3GNet-MP-2021.2.8-EFS ...")
model = M3GNet.load(CHECKPOINT)
pot = Potential(model=model)
calc = M3GNetCalculator(potential=pot)


def fcc(elem, a):
    return Structure(Lattice.cubic(a), [elem] * 4,
                     [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def bcc(elem, a):
    return Structure(Lattice.cubic(a), [elem] * 2,
                     [[0, 0, 0], [0.5, 0.5, 0.5]])


materials = [
    ("Cu (FCC)", fcc, 3.62126),
    ("Ni (FCC)", fcc, 3.5058),
    ("Mo (BCC)", bcc, 3.16762),
    ("Al (FCC)", fcc, 4.0396),
    ("Li (BCC)", bcc, 3.42682),
]


def energy_per_atom(s):
    atoms = AseAtomsAdaptor.get_atoms(s)
    atoms.calc = calc
    return float(atoms.get_potential_energy()) / len(s)


relaxer = Relaxer(potential=pot, relax_cell=True)

print("=" * 100)
print(f"{'Material':<12}{'offset':<8}{'a_init':>10}{'a_relax':>10}"
      f"{'E_dir, eV/at':>14}{'E_rel, eV/at':>14}"
      f"{'|dE_dir|, meV':>14}{'|dE_rel|, meV':>14}{'ratio':>8}")
print("=" * 100)

results = []
for name, builder, a_eq in materials:
    elem = name.split()[0]
    s_eq = builder(elem, a_eq)
    e_ref = energy_per_atom(s_eq)

    for offset_label, factor in [("+5%", 1.05), ("-5%", 0.95)]:
        a_off = a_eq * factor
        s_off = builder(elem, a_off)
        e_direct = energy_per_atom(s_off)
        res = relaxer.relax(s_off, fmax=0.02, steps=300, verbose=False)
        s_relaxed = res["final_structure"]
        e_relaxed = energy_per_atom(s_relaxed)
        a_relaxed = s_relaxed.lattice.abc[0]

        err_direct_meV = (e_direct - e_ref) * 1000
        err_relaxed_meV = (e_relaxed - e_ref) * 1000
        ratio = (abs(err_direct_meV) / abs(err_relaxed_meV)
                 if abs(err_relaxed_meV) > 1e-3 else float("inf"))

        print(f"{name:<12}{offset_label:<8}{a_off:>10.4f}{a_relaxed:>10.4f}"
              f"{e_direct:>14.4f}{e_relaxed:>14.4f}"
              f"{err_direct_meV:>+14.2f}{err_relaxed_meV:>+14.2f}"
              f"{ratio:>8.1f}")
        results.append({
            "material": name, "offset": offset_label,
            "a_MP": a_eq, "a_offset": a_off, "a_relaxed": a_relaxed,
            "E_ref_per_atom_eV": e_ref,
            "E_direct_per_atom_eV": e_direct,
            "E_relaxed_per_atom_eV": e_relaxed,
            "err_direct_meV": err_direct_meV,
            "err_relaxed_meV": err_relaxed_meV,
            "improvement_ratio": ratio,
        })
print("=" * 100)

ratios = [r["improvement_ratio"] for r in results
          if r["improvement_ratio"] != float("inf")]
err_direct = np.abs([r["err_direct_meV"] for r in results])
err_relaxed = np.abs([r["err_relaxed_meV"] for r in results])
mean_ratio = float(np.mean(ratios))
median_ratio = float(np.median(ratios))
min_ratio = float(np.min(ratios))
print(f"\nMean |dE_direct| = {np.mean(err_direct):.1f} meV/atom")
print(f"Mean |dE_relaxed| = {np.mean(err_relaxed):.2f} meV/atom")
print(f"Improvement ratio: mean = {mean_ratio:.1f}x, median = {median_ratio:.1f}x, "
      f"min = {min_ratio:.1f}x")
print(f"\nH2 criterion: min ratio >= 10x means confirmed")
verdict = bool(min_ratio >= 10)
print(f"H2 {'confirmed' if verdict else 'NOT confirmed'}")

out_path = SCRIPT_DIR / "expD_prerelax_results.json"
with open(out_path, "w") as f:
    json.dump({"model": "M3GNet-MP-2021.2.8-EFS",
               "results": results,
               "summary": {
                   "mean_err_direct_meV": float(np.mean(err_direct)),
                   "mean_err_relaxed_meV": float(np.mean(err_relaxed)),
                   "improvement_ratio_mean": mean_ratio,
                   "improvement_ratio_median": median_ratio,
                   "improvement_ratio_min": min_ratio,
                   "H2_confirmed": verdict,
               }}, f, indent=2)
print(f"\nSaved {out_path.name}")
