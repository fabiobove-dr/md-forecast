"""Build an offline 3Dmol.js viewer for the attributed README illustration."""

import argparse
import hashlib
import html
import json
from pathlib import Path
from urllib.request import urlopen

SOURCES = {
    "structure": (
        "https://files.rcsb.org/download/1HVR.pdb",
        "c8d3f238d3269a66823454d4681b71467b4677a18751ce094047961620b338cb",
    ),
    "library": (
        "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js",
        "f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016",
    ),
    "license": (
        "https://raw.githubusercontent.com/3dmol/3Dmol.js/2.5.5/LICENSE",
        "4c6eaaed856f3f28a3b1a98e74f4a8a71618de7d51ea4155c29f6f793bcef861",
    ),
}

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>1HVR protein–ligand illustration</title>
<style>
body{margin:0;background:#091322;color:#dce8ef;font:16px system-ui;
overflow-wrap:anywhere}
main{max-width:1400px;margin:auto;padding:20px}
#molecule{width:100%;height:720px;position:relative}
button{font:inherit;padding:10px 16px;margin:8px 8px 8px 0;cursor:pointer}
a{color:#7ee9bb}summary{cursor:pointer}pre{white-space:pre-wrap}
@media(max-width:600px){#molecule{height:420px}}
</style><main><h1>HIV-1 protease bound to inhibitor XK2 · PDB 1HVR</h1>
<p>Observed X-ray structure, shown as a static illustration. It is not a model
forecast or a benchmark trajectory. Drag to rotate; scroll to zoom.</p>
<div id="molecule" role="img"
aria-label="Interactive HIV protease dimer and ligand XK2"></div>
<p id="status" role="status">Loading the embedded structure…</p>
<button id="reset" disabled>Reset view</button>
<button id="save" disabled>Save molecular image (PNG)</button>
<p>Protein chains: teal / blue; ligand: gold carbon, blue nitrogen, red oxygen.
The translucent green surface shows nearby protein atoms, not an inferred field.</p>
<p>Structure: <a href="https://www.rcsb.org/structure/1HVR">RCSB PDB 1HVR</a>,
Lam et al., Science (1994), doi:10.1126/science.8278812.
PDB data: CC0. Renderer: <a href="https://3dmol.org">3Dmol.js</a> 2.5.5.</p>
<details><summary>Source hashes and renderer license</summary>
<pre>__PROVENANCE__</pre><pre>__LICENSE__</pre></details></main>
<script>__LIBRARY__</script><script>
const pdb=__STRUCTURE__;
async function draw(){
  try{
    $3Dmol.setSyncSurface(true);
    const viewer=$3Dmol.createViewer('molecule',
      {backgroundColor:'#091322',antialias:true});
    const model=viewer.addModel(pdb,'pdb');
    const protein={resn:'XK2',invert:true};
    const ligand={resn:'XK2'};
    viewer.setStyle({},{cartoon:{color:'#4baaa5'}});
    viewer.setStyle({chain:'B'},{cartoon:{color:'#76a8d1'}});
    viewer.setStyle(ligand,{stick:{radius:0.24,colorscheme:'yellowCarbon'},
      sphere:{scale:0.28,colorscheme:'yellowCarbon'}});
    const nearby=model.selectedAtoms({and:[protein,{within:{distance:5,sel:ligand}}]});
    viewer.addStyle({index:nearby.map(a=>a.index)},{stick:{radius:0.11,color:'#9cbdb9'}});
    await viewer.addSurface($3Dmol.SurfaceType.VDW,{opacity:0.12,color:'#78c6a4'},
      {index:nearby.map(a=>a.index)});
    viewer.zoomTo();viewer.rotate(90,'y');viewer.zoom(1.22);viewer.render();
    const initial=viewer.getView();
    window.moleculeViewer=viewer;
    document.getElementById('reset').onclick=()=>{viewer.setView(initial);viewer.render()};
    document.getElementById('save').onclick=()=>{
      viewer.render();const a=document.createElement('a');a.href=viewer.pngURI();
      a.download='protein-ligand-complex.png';a.click();
    };
    document.getElementById('reset').disabled=false;
    document.getElementById('save').disabled=false;
    document.getElementById('status').textContent=
      'Embedded structure ready; no network connection is required.';
  }catch(error){document.getElementById('status').textContent='3D view unavailable. '
    +'Open the README PNG for the static illustration. '+error.message;}
}
draw();
</script></html>"""


def main() -> None:
    """Fetch verified public inputs and write a self-contained local viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="HTML output in an ignored directory")
    output = parser.parse_args().output
    payloads = {}
    for name, (url, digest) in SOURCES.items():
        with urlopen(url, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(
                f"{name} checksum changed; review the source before rendering"
            )
        payloads[name] = data.decode("utf-8")
    page = PAGE.replace("__PROVENANCE__", html.escape(json.dumps(SOURCES, indent=2)))
    page = page.replace("__LICENSE__", html.escape(payloads["license"]))
    page = page.replace(
        "__LIBRARY__", payloads["library"].replace("</script", r"<\/script")
    )
    page = page.replace(
        "__STRUCTURE__", json.dumps(payloads["structure"]).replace("<", r"\u003c")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
