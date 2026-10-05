# Molecular illustrations

The README image is a 3Dmol.js rendering of the observed X-ray structure
[PDB 1HVR](https://www.rcsb.org/structure/1HVR): HIV-1 protease bound to the
cyclic-urea inhibitor XK2. Structure authors: Lam et al., *Science* (1994),
[doi:10.1126/science.8278812](https://doi.org/10.1126/science.8278812).
It is a static experimental illustration, not a model forecast or a trajectory
from the forecasting benchmark.

The coordinates come from the [PDB archive](https://files.rcsb.org/download/1HVR.pdb),
whose data files are [CC0](https://www.rcsb.org/pages/policies).
The renderer is [3Dmol.js](https://3dmol.org/doc/) **2.5.5**; the complete
[upstream license](https://github.com/3dmol/3Dmol.js/blob/2.5.5/LICENSE)
is retained in every generated HTML viewer. Source URLs and SHA-256 checksums
are pinned in [the reproduction example](https://github.com/fabiobove-dr/md-forecast/blob/main/examples/render_readme.py).
An upstream byte change fails verification and requires a deliberate review.
Only the derived README PNG is committed; coordinate/library downloads and
generated HTML viewers remain outside tracked files.

## Reproduce the image and offline viewer

From the checkout root:

```sh
uv run --locked python examples/render_readme.py data/reports/readme-3dmol/viewer.html
```

The command downloads the three checksum-verified public inputs (structure,
renderer and license). Open the output HTML in a browser. It embeds all inputs;
viewing requires no CDN, server or network connection. Drag to rotate, scroll
to zoom, use **Reset view**, and export with **Save molecular image (PNG)**.
GitHub READMEs display the exported PNG; the interactive viewer is a separate
local artifact.

The reviewed export used a 1,400 × 720 CSS-pixel molecular container in desktop
Chrome. Raster dimensions/antialiasing can vary by browser and graphics device;
the molecular selections and initial camera are fixed in the example:

- Protein chains A/B: teal/blue cartoon ribbons.
- Ligand: residue `XK2`, sticks and small spheres, with yellow carbon and
  element-colored heteroatoms.
- Nearby protein atoms: within 5 Å of ligand atoms, shown as thin sticks and
  a translucent green van der Waals surface. This is a visualization selection,
  not the benchmark's contact/pocket definition or an inferred physical field.
- Camera: fit all atoms, rotate 90 degrees about y, then zoom by 1.22.

## Forecast reports

[Issue #44](https://github.com/fabiobove-dr/md-forecast/issues/44) tracks packaging
the saved-result HTML report with inline 3Dmol.js, verified structure provenance,
exact declared regions, PNG export and a static fallback when WebGL is absent.
Reports must distinguish an illustrative structure, an observed reference and
measured trajectory coordinates. Scalar observable forecasts must not be
displayed as invented atomistic motion.
