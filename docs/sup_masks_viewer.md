WP5 Sup-Masks 3D Viewer (Streamlit + PyVista)

Overview
- A standalone Streamlit app to visualize supervision masks, labels, seeds and unlabeled voxels in 3D.
- Built on PyVista and streamlit-pyvista, with headless support via Xvfb.
- Sup-masks directory is locked to the run folder used for experiments.

Install (inside your venv)
- venv/bin/python -m pip install -U streamlit pyvista vtk streamlit-pyvista nibabel "monai[all]" "pyvista[trame]" trame trame-vtk
- sudo apt-get update && sudo apt-get install -y xvfb

Run
- venv/bin/python -m streamlit run scripts/vis_sup_masks_streamlit.py -- --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset --datalist datalist_train.json

Locked directories
- Sup-masks: /home/peisheng/MONAI/runs/sup_masks_0p1pct_global_d0_5_nov
- SV fill A: /home/peisheng/MONAI/runs/sv_fill_0p1pct
- SV fill B: /home/peisheng/MONAI/runs/sv_fill_0p1pct_n10k
- To change these, edit the constants near the top of `main()` in scripts/vis_sup_masks_streamlit.py.

Controls (sidebar)
- Show image volume: toggle 3D volume (heavy; leave off for speed)
- Show label surfaces: iso-surfaces for classes 0..4 (class 6 ignored)
- Show supervised region: green iso-surface of the dilated supervision region
- Show seed points: colored seed voxels per class
- Show unlabeled (sampled points): gray random sample of unlabeled voxels
- Classes (0..4): filter which classes to display for surfaces and seeds
- Surface decimation: 0.0..0.9; fraction of triangles to remove (see notes below)
- Surface downsample (stride): 1..4; higher is faster; set 1 for exact overlay with volume
- Unlabeled sample (points): cap number of unlabeled points drawn

Legends
- Sup/Seeds tab: classes + supervised region
- SV Fill tab: classes
- SV Compare tab: classes; seeds panel uses GT label colors

Performance tips
- Start with: volume OFF, label surfaces OFF, supervised ON, seeds ON
- For guaranteed surfaces: set decimation=0.0 and stride=1 (heavier)
- Typical preview: decimation=0.4–0.6, stride=2–3
- Reduce unlabeled points to 5k–10k
- For exact overlay with volume, set stride=1 (heavier)

Troubleshooting
- Headless display: the app starts Xvfb automatically (pv.start_xvfb). If missing: sudo apt-get install -y xvfb
- WebSocketClosedError: the app falls back to a static screenshot when the interactive bridge fails
- Bad X server connection: ensure Xvfb is present, or run on a machine with DISPLAY

Key code references
- Main app: scripts/vis_sup_masks_streamlit.py
- 3D embed (stpyvista): scripts/vis_sup_masks_streamlit.py
- Default camera framing: scripts/vis_sup_masks_streamlit.py
- Class colors: scripts/vis_sup_masks_streamlit.py
- Class filtering & surfaces: scripts/vis_sup_masks_streamlit.py

SV Compare tab (A vs B)
- Three 3D plots: Seeds | A surfaces | B surfaces
- 2D SV-ID slices: A and B side-by-side
- Present classes readout: counts for classes 0..4 for A/B
- Debug expander shows shapes to verify alignment
- Tips if surfaces are missing:
  - Ensure selected classes exist in the readout
  - Set stride=1 and decimation=0.0
  - Turn off volume overlay to check surfaces alone

Surface decimation details
- Uses PyVista `decimate_pro` with a reduction fraction (0.0–0.9)
- 0.0 disables simplification; preserves thin structures
- Higher values remove more triangles; faster but can erase small parts
- The app keeps the original mesh if decimation removes too many points
