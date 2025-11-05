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

Locked sup_masks directory
- The app reads masks from a fixed folder by default: /home/peisheng/MONAI/runs/sup_masks_0p1pct_global_d0_5_nov
- To change it, edit SUP_DIR_DEFAULT in scripts/vis_sup_masks_streamlit.py:434

Controls (sidebar)
- Show image volume: toggle 3D volume (heavy; leave off for speed)
- Show label surfaces: iso-surfaces for classes 0..4 (class 6 ignored)
- Show supervised region: green iso-surface of the dilated supervision region
- Show seed points: colored seed voxels per class
- Show unlabeled (sampled points): gray random sample of unlabeled voxels
- Classes (0..4): filter which classes to display for surfaces and seeds
- Surface decimation: 0.0..0.9; higher is faster, fewer polygons
- Surface downsample (stride): 1..4; higher is faster; set 1 for exact overlay with volume
- Unlabeled sample (points): cap number of unlabeled points drawn

Legend
- Always shown under the 3D view with colors for:
  - Class 0 (bg) #b0b0b0
  - Class 1 #ff3b30
  - Class 2 #ff9500
  - Class 3 #007aff
  - Class 4 #5856d6
  - Supervised #34c759

Performance tips
- Start with: volume OFF, label surfaces OFF, supervised ON, seeds ON
- Use decimation 0.6–0.8 and stride 3–4 for quick previews
- Reduce unlabeled points to 5k–10k
- For exact overlay with volume, set stride=1 (heavier)

Troubleshooting
- Headless display: the app starts Xvfb automatically (pv.start_xvfb). If missing: sudo apt-get install -y xvfb
- WebSocketClosedError: the app falls back to a static screenshot when the interactive bridge fails
- Bad X server connection: ensure Xvfb is present, or run on a machine with DISPLAY

Key code references
- Locked sup dir: scripts/vis_sup_masks_streamlit.py:434
- 3D embed (stpyvista): scripts/vis_sup_masks_streamlit.py:541
- Default camera framing: scripts/vis_sup_masks_streamlit.py:251
- Class colors: scripts/vis_sup_masks_streamlit.py:287
- Class filtering: scripts/vis_sup_masks_streamlit.py:314, scripts/vis_sup_masks_streamlit.py:333

