import os
import time
import warnings
import json
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from m3gnet.models import M3GNet, Potential, M3GNetCalculator
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get("M3GNET_CHECKPOINT", "MP-2021.2.8-EFS")
MODEL_NAME = "M3GNet-MP-2021.2.8-EFS"
print(f"Loading {MODEL_NAME} ...")
t_load = time.time()
model = M3GNet.load(CHECKPOINT)
pot = Potential(model=model)
calc = M3GNetCalculator(potential=pot)
load_time = time.time() - t_load
print(f"  loaded in {load_time:.2f} s")

structures = {
    "Si (diamond)": Structure(
        Lattice.cubic(5.469),
        ["Si"] * 8,
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.0, 0.5, 0.5], [0.25, 0.75, 0.75],
         [0.5, 0.0, 0.5], [0.75, 0.25, 0.75], [0.5, 0.5, 0.0], [0.75, 0.75, 0.25]],
    ),
    "Cu (FCC)": Structure(
        Lattice.cubic(3.621),
        ["Cu"] * 4,
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    ),
    "Ni (FCC)": Structure(
        Lattice.cubic(3.506),
        ["Ni"] * 4,
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    ),
    "LiF (halite)": Structure(
        Lattice.cubic(4.083),
        ["Li", "Li", "Li", "Li", "F", "F", "F", "F"],
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
         [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
    ),
    "Mo (BCC)": Structure(
        Lattice.cubic(3.168),
        ["Mo", "Mo"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    ),
}

results = []
print("\n" + "=" * 80)
print(f"{'Structure':<20}{'N atoms':>8}{'E [eV]':>15}{'E [eV/atom]':>15}{'|f|max':>12}{'t [s]':>10}")
print("=" * 80)
for name, s in structures.items():
    atoms = AseAtomsAdaptor.get_atoms(s)
    atoms.calc = calc
    _ = atoms.get_potential_energy()
    t0 = time.time()
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces())
    dt = time.time() - t0
    e_per_atom = e / len(s)
    fmax = float(np.linalg.norm(f, axis=1).max())
    print(f"{name:<20}{len(s):>8}{e:>15.4f}{e_per_atom:>15.4f}{fmax:>12.4f}{dt:>10.3f}")
    results.append({
        "name": name,
        "n_atoms": len(s),
        "formula": s.composition.reduced_formula,
        "lattice_a": s.lattice.abc[0],
        "energy_eV": e,
        "energy_eV_per_atom": e_per_atom,
        "fmax_eV_per_A": fmax,
        "inference_time_s": dt,
    })
print("=" * 80)

out_path = SCRIPT_DIR / "expA_results.json"
with open(out_path, "w") as f:
    json.dump({"model": MODEL_NAME, "checkpoint": CHECKPOINT,
               "load_time_s": load_time, "results": results}, f, indent=2)
print(f"Saved {out_path.name}")
