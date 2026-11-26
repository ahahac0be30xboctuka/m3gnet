# Stress-testing M3GNet near equilibrium

An independent experimental study of a pretrained interatomic potential. **How does M3GNet respond to lattice strain, and how close do relaxed lattices stay to their references across bonding classes?** I designed two comparisons that address these questions while keeping the published `M3GNet-MP-2021.2.8-EFS` checkpoint fixed. [Original M3GNet paper](https://doi.org/10.1038/s43588-022-00349-3)

[Experiment protocol](scripts/README.md) · [Lattice results](scripts/expC_classes_results.json) · [Strain results](scripts/expD_prerelax_results.json)

## Findings

### Cubic lattices stayed close to their reference values

I relaxed 12 cubic crystals from their Materials Project lattice parameters and measured how far each lattice moved. Metals and covalent crystals had similar mean deviations, and ionic crystals had the smallest mean on this panel.

| Bonding class | Crystals | Mean relative lattice deviation | Largest deviation |
| :--- | ---: | ---: | ---: |
| Metallic | 4 | 0.129% | 0.245% |
| Covalent | 4 | 0.136% | 0.353% |
| Ionic | 4 | 0.072% | 0.123% |

This panel gives no clear support for the hypothesis that metals stay closer to their reference lattices. Each relaxation starts at the reference lattice, so the comparison probes local behavior on four examples per class. [Per-crystal data and Materials Project IDs](scripts/expC_classes_results.json)

### Relaxation removed most of the energy deviation induced by strain

I applied ±5% isotropic lattice strain to Cu, Ni, Mo, Al, and Li. For each of the ten strained cells, I compared the predicted energy before and after relaxation with the energy that **the same checkpoint** assigns to the reference geometry.

| Geometry scored by M3GNet | Mean absolute energy deviation |
| :--- | ---: |
| Strained cell | 136.73 meV/atom |
| Relaxed cell | 0.085 meV/atom |

The reduction exceeded the proposed 10× threshold in all eight cases with a finite ratio. In the other two cases, both for Mo, the recorded residual was zero at the precision of the run. The result quantifies recovery of the model's reference energy within this strain sweep. Predictive accuracy would require independent energy references at matched geometries. [Per-material results and summary](scripts/expD_prerelax_results.json)

## Experimental controls

- **Force check.** Five symmetric reference cells produced maximum atomic force norms of roughly 10⁻⁸–10⁻⁷ eV/Å. Symmetry can give near-zero net forces even when the lattice parameter needs to relax. [Forces and energies](scripts/expA_results.json)
- **Recovery from a displaced geometry.** Variable-cell relaxation took stretched BCC molybdenum from 3.300 Å to 3.16788 Å, within 0.0081% of the 3.16762 Å Materials Project reference. A displaced 16-atom supercell provides a second recovery case. [Molybdenum results](scripts/expB_results.json)

## Scope

The class comparison uses selected cubic crystals near equilibrium. It leaves performance on unseen materials, low-symmetry structures and larger deformations open. The timing stored in experiment A follows an untimed call on the same structure, so it can include cache hits. A meaningful latency benchmark would time fresh calculations after a separate model warm-up. [Experiment A code](scripts/expA_predict.py#L60-L69)

## Reproduce the experiments

The four Python scripts run independently and each writes a JSON result alongside its source. The archived outputs were collected on Apple Silicon using Python 3.10, TensorFlow 2.13 and M3GNet 0.2.4. [Environment, commands and metric definitions](scripts/README.md)
