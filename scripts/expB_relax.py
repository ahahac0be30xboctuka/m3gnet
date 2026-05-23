import os
import time
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
MODEL_NAME = "M3GNet-MP-2021.2.8-EFS"
print(f"Loading {MODEL_NAME} ...")
model = M3GNet.load(CHECKPOINT)
pot = Potential(model=model)
calc = M3GNetCalculator(potential=pot)

init = Structure(Lattice.cubic(3.3), ["Mo", "Mo"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
print("\n=== Test case: Mo BCC stretched (a=3.3 A, target a~3.168 A) ===")
print(f"Initial lattice a: {init.lattice.abc[0]:.4f} A")
print(f"Initial volume:    {init.volume:.4f} A^3")

a0 = AseAtomsAdaptor.get_atoms(init)
a0.calc = calc
e_init = float(a0.get_potential_energy())
fmax_init = float(np.linalg.norm(a0.get_forces(), axis=1).max())
print(f"Initial energy:    {e_init:.4f} eV  ({e_init/len(init):.4f} eV/atom)")
print(f"Initial fmax:      {fmax_init:.4f} eV/A")

relaxer = Relaxer(potential=pot, relax_cell=True)
t0 = time.time()
res = relaxer.relax(init, fmax=0.01, steps=500, verbose=True)
elapsed = time.time() - t0

final = res["final_structure"]
traj = res["trajectory"]
e_final_total = float(traj.energies[-1])
e_final_per_atom = e_final_total / len(final)
n_steps = len(traj.energies)

print(f"\nRelaxation finished:")
print(f"  steps:           {n_steps}")
print(f"  wall time:       {elapsed:.2f} s")
print(f"  final lattice a: {final.lattice.abc[0]:.4f} A (MP ~ 3.16762 A)")
print(f"  final volume:    {final.volume:.4f} A^3")
print(f"  final energy:    {e_final_total:.4f} eV  ({e_final_per_atom:.4f} eV/atom)")
print(f"  dE total:        {e_final_total - e_init:+.4f} eV")
print(f"  MP reference:    -10.8456 eV/atom (mp-129)")

print("\n=== Test case 2: Mo 2x2x2 supercell, perturbed 0.1 A ===")
init2 = init.copy()
init2.make_supercell([2, 2, 2])
np.random.seed(42)
init2.perturb(0.1)
print(f"  N atoms:         {len(init2)}")
print(f"  Initial volume:  {init2.volume:.4f} A^3")
a02 = AseAtomsAdaptor.get_atoms(init2)
a02.calc = calc
e_init2 = float(a02.get_potential_energy())
fmax_init2 = float(np.linalg.norm(a02.get_forces(), axis=1).max())
print(f"  Initial energy:  {e_init2:.4f} eV  ({e_init2/len(init2):.4f} eV/atom)")
print(f"  Initial fmax:    {fmax_init2:.4f} eV/A")

t0 = time.time()
res2 = relaxer.relax(init2, fmax=0.01, steps=500, verbose=False)
elapsed2 = time.time() - t0
final2 = res2["final_structure"]
traj2 = res2["trajectory"]
e_final2_total = float(traj2.energies[-1])
e_final2_pa = e_final2_total / len(final2)
n_steps2 = len(traj2.energies)
print(f"  steps:           {n_steps2}")
print(f"  wall time:       {elapsed2:.2f} s")
print(f"  final energy:    {e_final2_pa:.4f} eV/atom")
print(f"  dE/atom:         {e_final2_pa - e_init2/len(init2):+.4f} eV/atom")

out_path = SCRIPT_DIR / "expB_results.json"
with open(out_path, "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "checkpoint": CHECKPOINT,
        "test1_Mo_stretched": {
            "n_atoms": len(init),
            "a_init_A": init.lattice.abc[0],
            "a_final_A": final.lattice.abc[0],
            "a_mp_ref_A": 3.16762,
            "e_init_eV_per_atom": e_init / len(init),
            "e_final_eV_per_atom": e_final_per_atom,
            "fmax_init_eV_per_A": fmax_init,
            "steps": n_steps,
            "time_s": elapsed,
        },
        "test2_Mo_supercell_perturbed": {
            "n_atoms": len(init2),
            "e_init_eV_per_atom": e_init2 / len(init2),
            "e_final_eV_per_atom": e_final2_pa,
            "fmax_init_eV_per_A": fmax_init2,
            "steps": n_steps2,
            "time_s": elapsed2,
        },
    }, f, indent=2)
print(f"\nSaved {out_path.name}")
