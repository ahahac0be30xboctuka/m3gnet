import os
import time
import warnings
import json
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from m3gnet.models import M3GNet, Potential, Relaxer
from pymatgen.core import Structure, Lattice

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get("M3GNET_CHECKPOINT", "MP-2021.2.8-EFS")
print("Loading M3GNet-MP-2021.2.8-EFS ...")
model = M3GNet.load(CHECKPOINT)
pot = Potential(model=model)


def fcc(elem, a):
    return Structure(Lattice.cubic(a), [elem] * 4,
                     [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def bcc(elem, a):
    return Structure(Lattice.cubic(a), [elem] * 2,
                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def diamond(elem, a):
    return Structure(Lattice.cubic(a), [elem] * 8,
                     [[0, 0, 0], [0.25, 0.25, 0.25], [0, 0.5, 0.5], [0.25, 0.75, 0.75],
                      [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0.5, 0.5, 0], [0.75, 0.75, 0.25]])


def zincblende(elem_a, elem_b, a):
    return Structure(Lattice.cubic(a), [elem_a, elem_a, elem_a, elem_a, elem_b, elem_b, elem_b, elem_b],
                     [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
                      [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]])


def rocksalt(cation, anion, a):
    return Structure(Lattice.cubic(a), [cation, cation, cation, cation, anion, anion, anion, anion],
                     [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
                      [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])


materials = [
    ("metal",    "Cu (FCC)",   "mp-30",    3.62126, lambda a: fcc("Cu", a)),
    ("metal",    "Ni (FCC)",   "mp-23",    3.5058,  lambda a: fcc("Ni", a)),
    ("metal",    "Mo (BCC)",   "mp-129",   3.16762, lambda a: bcc("Mo", a)),
    ("metal",    "Al (FCC)",   "mp-134",   4.0396,  lambda a: fcc("Al", a)),
    ("covalent", "Si (dia)",   "mp-149",   5.46873, lambda a: diamond("Si", a)),
    ("covalent", "Ge (dia)",   "mp-32",    5.7677,  lambda a: diamond("Ge", a)),
    ("covalent", "C (dia)",    "mp-66",    3.5687,  lambda a: diamond("C", a)),
    ("covalent", "SiC (zb)",   "mp-8062",  4.3815,  lambda a: zincblende("Si", "C", a)),
    ("ionic",    "LiF (rs)",   "mp-1138",  4.0830,  lambda a: rocksalt("Li", "F", a)),
    ("ionic",    "NaCl (rs)",  "mp-22862", 5.6920,  lambda a: rocksalt("Na", "Cl", a)),
    ("ionic",    "MgO (rs)",   "mp-1265",  4.2520,  lambda a: rocksalt("Mg", "O", a)),
    ("ionic",    "KCl (rs)",   "mp-23193", 6.3940,  lambda a: rocksalt("K", "Cl", a)),
]

relaxer = Relaxer(potential=pot, relax_cell=True)
print("=" * 75)
print(f"{'Class':<10}{'Material':<14}{'mp-id':<12}{'a_MP, A':>10}{'a_relx, A':>12}{'|da|/a, %':>12}")
print("=" * 75)

results = []
for cls, name, mpid, a_mp, builder in materials:
    s = builder(a_mp)
    t0 = time.time()
    res = relaxer.relax(s, fmax=0.02, steps=200, verbose=False)
    dt = time.time() - t0
    a_relaxed = res["final_structure"].lattice.abc[0]
    rel_dev = abs(a_relaxed - a_mp) / a_mp * 100
    print(f"{cls:<10}{name:<14}{mpid:<12}{a_mp:>10.4f}{a_relaxed:>12.4f}{rel_dev:>12.3f}")
    results.append({
        "class": cls, "material": name, "mp_id": mpid,
        "a_MP": a_mp, "a_relaxed": a_relaxed,
        "rel_dev_percent": rel_dev,
        "abs_dev_A": abs(a_relaxed - a_mp),
        "time_s": dt,
    })
print("=" * 75)

classes = ["metal", "covalent", "ionic"]
print(f"\n{'Class':<12}{'mean |da|/a, %':>18}{'std, %':>10}{'max, %':>10}{'N':>4}")
print("-" * 55)
summary = {}
for cls in classes:
    devs = [r["rel_dev_percent"] for r in results if r["class"] == cls]
    summary[cls] = {
        "mean_rel_dev_pct": float(np.mean(devs)),
        "std_rel_dev_pct": float(np.std(devs)),
        "max_rel_dev_pct": float(np.max(devs)),
        "N": len(devs),
    }
    print(f"{cls:<12}{summary[cls]['mean_rel_dev_pct']:>18.3f}"
          f"{summary[cls]['std_rel_dev_pct']:>10.3f}{summary[cls]['max_rel_dev_pct']:>10.3f}"
          f"{summary[cls]['N']:>4}")
print("-" * 55)

mean_metal = summary["metal"]["mean_rel_dev_pct"]
mean_cov = summary["covalent"]["mean_rel_dev_pct"]
mean_ion = summary["ionic"]["mean_rel_dev_pct"]
ratio_cov = mean_cov / mean_metal if mean_metal > 0 else float("inf")
ratio_ion = mean_ion / mean_metal if mean_metal > 0 else float("inf")
print(f"\nH1: ratio covalent/metal = {ratio_cov:.2f}, ionic/metal = {ratio_ion:.2f}")
print(f"H1 criterion: ratio >= 2.0 means confirmed for that class")
verdict = []
if ratio_cov >= 2.0:
    verdict.append("covalent")
if ratio_ion >= 2.0:
    verdict.append("ionic")
if verdict:
    print(f"H1 confirmed for classes: {', '.join(verdict)}")
else:
    print("H1 NOT confirmed (or effect smaller than 2x)")

out_path = SCRIPT_DIR / "expC_classes_results.json"
with open(out_path, "w") as f:
    json.dump({"model": "M3GNet-MP-2021.2.8-EFS",
               "materials": results,
               "summary_by_class": summary,
               "ratios": {"covalent_over_metal": ratio_cov, "ionic_over_metal": ratio_ion},
               "H1_verdict": verdict}, f, indent=2)
print(f"\nSaved {out_path.name}")
