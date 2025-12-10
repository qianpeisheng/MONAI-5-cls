#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web frontend for the detection+segmentation pipeline
"""

import fnmatch
import glob
import io
import os
import time
import yaml
from contextlib import contextmanager
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from stpyvista import stpyvista

# Configurable simulation delay settings (in seconds)
# Modify these values to control the simulated processing times
SIMULATION_DELAYS = {
    "conversion": 1.0,      # Base delay for NII to JPG conversion
    "detection": 1.0,       # Base delay for object detection
    "bbox": 1.0,            # Base delay for 3D bounding box generation
    "segmentation": 1.0,    # Base delay for 3D segmentation
    "metrology": 1.0,       # Base delay for metrology analysis
    "progress_steps": 1,    # Number of progress steps to show
    "progress_delay": 0.1   # Delay between progress steps
}

# Enable off-screen rendering in headless environments
try:
    pv.start_xvfb()
except:
    pass

@contextmanager
def timer(section_name):
    """Context manager for timing code blocks."""
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    st.write(f"[{section_name}] completed in {elapsed_time:.2f} seconds.")


def simulate_conversion(output_path, progress_callback=None):
    """Simulate the conversion from 3D NII to 2D images."""
    # In a real scenario, this would convert NII to JPG
    # Here we just pretend to work and return the path to pre-generated images
    time.sleep(SIMULATION_DELAYS["conversion"])  # Simulate base processing time

    if progress_callback:
        steps = SIMULATION_DELAYS["progress_steps"]
        step_delay = SIMULATION_DELAYS["progress_delay"]
        for i in range(steps):
            time.sleep(step_delay)
            progress_callback(i/steps)

    return os.path.join(output_path, "view1", "input_images"), os.path.join(output_path, "view2", "input_images")


def simulate_object_detection(output_path, progress_callback=None):
    """Simulate the object detection process."""
    # In a real scenario, this would run inference
    # Here we just pretend to work and return the pre-generated detections
    time.sleep(SIMULATION_DELAYS["detection"])  # Simulate base processing time

    if progress_callback:
        steps = SIMULATION_DELAYS["progress_steps"]
        step_delay = SIMULATION_DELAYS["progress_delay"]
        for i in range(steps):
            time.sleep(step_delay)
            progress_callback(i/steps)

    return os.path.join(output_path, "view1", "detections", "visualize"), os.path.join(output_path, "view2", "detections", "visualize")


def simulate_3d_bbox_generation(output_path, progress_callback=None):
    """Simulate the 3D bounding box generation process."""
    # In a real scenario, this would compute 3D bboxes from 2D detections
    time.sleep(SIMULATION_DELAYS["bbox"])  # Simulate base processing time

    if progress_callback:
        steps = SIMULATION_DELAYS["progress_steps"]
        step_delay = SIMULATION_DELAYS["progress_delay"]
        for i in range(steps):
            time.sleep(step_delay)
            progress_callback(i/steps)

    return os.path.join(output_path, "bb3d.npy")


def simulate_3d_segmentation(output_path, bb2d_config, progress_callback=None):
    """Simulate the 3D segmentation process."""
    # In a real scenario, this would perform 3D segmentation
    time.sleep(SIMULATION_DELAYS["segmentation"])  # Simulate base processing time

    if progress_callback:
        steps = SIMULATION_DELAYS["progress_steps"]
        step_delay = SIMULATION_DELAYS["progress_delay"]
        for i in range(steps):
            time.sleep(step_delay)
            progress_callback(i/steps)

    # Get the specific segmentation files from bb2d.yaml
    good_seg_file = bb2d_config.get("good")
    bad_seg_file = bb2d_config.get("bad")

    if not good_seg_file or not bad_seg_file:
        raise ValueError("Both 'good' and 'bad' segmentation files must be specified in bb2d.yaml")

    # Paths to the bump-level segmentation files
    good_seg_path = os.path.join(output_path, "mmt", "pred", "class_0", good_seg_file)
    bad_seg_path = os.path.join(output_path, "mmt", "pred", "class_0", bad_seg_file)

    # Check if segmentation files exist
    if not os.path.exists(good_seg_path):
        raise FileNotFoundError(f"Good segmentation file not found: {good_seg_path}")
    if not os.path.exists(bad_seg_path):
        raise FileNotFoundError(f"Bad segmentation file not found: {bad_seg_path}")

    return good_seg_path, bad_seg_path


def simulate_metrology(output_path, progress_callback=None):
    """Simulate the metrology analysis process."""
    # In a real scenario, this would compute metrology information
    time.sleep(SIMULATION_DELAYS["metrology"])  # Simulate base processing time

    if progress_callback:
        steps = SIMULATION_DELAYS["progress_steps"]
        step_delay = SIMULATION_DELAYS["progress_delay"]
        for i in range(steps):
            time.sleep(step_delay)
            progress_callback(i/steps)

    # Check for existing metrology files
    metrology_dir = os.path.join(output_path, "metrology")
    csv_path = os.path.join(metrology_dir, "memory.csv")

    # Try to find final_report.pdf first, then fall back to memory_report.pdf
    final_pdf_path = os.path.join(metrology_dir, "final_report.pdf")
    memory_pdf_path = os.path.join(metrology_dir, "memory_report.pdf")

    # Select the first available PDF report
    if os.path.exists(final_pdf_path):
        pdf_path = final_pdf_path
    else:
        pdf_path = memory_pdf_path

    # Check if required files exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrology CSV file not found: {csv_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Metrology PDF report not found: {pdf_path}")

    return csv_path, pdf_path


def load_bb2d_config(selected_sample):
    """Load the bb2d.yaml configuration file for the sample."""
    config_path = os.path.join("output/demo", selected_sample, "bb2d.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            st.warning(f"Error loading bb2d.yaml: {str(e)}")
    return {
        "view1": None,
        "view2": None,
        "good": None,
        "bad": None
    }  # Default if file doesn't exist or has issues


def calculate_crop_dimensions(sample_path, bb2d_config):
    """Calculate common dimensions for center cropping both views."""
    # Get paths to the specified images
    view1_image_path = os.path.join(sample_path, "view1", "input_images", bb2d_config["view1"]) if bb2d_config["view1"] else None
    view2_image_path = os.path.join(sample_path, "view2", "input_images", bb2d_config["view2"]) if bb2d_config["view2"] else None

    # Default dimensions if we can't determine from actual images
    min_width, min_height = 600, 600

    # Try to get dimensions from actual images
    if view1_image_path and os.path.exists(view1_image_path):
        img1 = Image.open(view1_image_path)
        width1, height1 = img1.size

        if view2_image_path and os.path.exists(view2_image_path):
            img2 = Image.open(view2_image_path)
            width2, height2 = img2.size

            # Find minimum dimensions
            min_width = min(width1, width2)
            min_height = min(height1, height2)
        else:
            min_width, min_height = width1, height1
    elif view2_image_path and os.path.exists(view2_image_path):
        img2 = Image.open(view2_image_path)
        min_width, min_height = img2.size

    return min_width, min_height


def center_crop_image(image_path, target_width, target_height):
    """Center crop the image to the specified dimensions."""
    if not os.path.exists(image_path):
        return None

    try:
        # Open the image
        img = Image.open(image_path)
        width, height = img.size

        # Calculate crop coordinates for center crop
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img
    except Exception as e:
        st.error(f"Error cropping image {image_path}: {str(e)}")
        return None


def display_image_file(image_path, target_width=None, target_height=None):
    """Display an image directly from the given file path, with optional cropping."""
    if os.path.exists(image_path):
        # Extract filename without path
        filename = os.path.basename(image_path)

        # Convert "imageXXX.jpg" to "Frame XXX"
        if filename.startswith("image") and filename.endswith(".jpg"):
            try:
                # Extract the number part from imageXXX.jpg
                frame_number = filename.replace("image", "").replace(".jpg", "")
                caption = f"Frame {frame_number}"
            except:
                caption = filename
        else:
            caption = filename

        # If target dimensions are provided, crop the image
        if target_width and target_height:
            cropped_img = center_crop_image(image_path, target_width, target_height)
            if cropped_img:
                st.image(cropped_img, caption=caption, use_container_width=True)
            else:
                st.error(f"Failed to crop image: {image_path}")
        else:
            # Display the image without cropping
            st.image(image_path, caption=caption, use_container_width=True)
    else:
        st.error(f"Image file not found: {image_path}")


def display_detection_images_with_slider(detection_path, view_label):
    """Display detection images with a slider to browse through them."""
    image_files = sorted(glob.glob(os.path.join(detection_path, "*.jpg")))
    if image_files:
        # Create a slider to navigate through images
        num_images = len(image_files)
        selected_idx = st.slider(f"Select {view_label} Image", 0, num_images-1, num_images//2, key=f"slider_{view_label}")

        # Display the selected image
        selected_image = image_files[selected_idx]
        st.image(selected_image, caption=f"{view_label} Detection (Image {selected_idx+1}/{num_images})", use_container_width=True)
    else:
        st.error(f"No detection visualization images found for {view_label}")


def display_segmentation_point_cloud(segmentation_path, is_full_scan=True):
    """Display a 3D point cloud visualization of segmentation using PyVista."""
    if not os.path.exists(segmentation_path):
        st.error(f"Segmentation file not found: {segmentation_path}")
        return

    try:
        # Load segmentation data
        img = nib.load(segmentation_path)
        data = img.get_fdata()
        affine = img.affine

        # Set parameters based on full scan or bump level
        if is_full_scan:
            title = "Full-Scan 3D Segmentation"
            keep_ratio = 0.1  # Downsample for full scan (larger)
            key_suffix = "full_scan"
        else:
            title = "Bump-Level 3D Segmentation"
            keep_ratio = 1.0   # No downsampling for bump level (smaller)
            key_suffix = "bump_level"

        # Hardcoded visualization parameters
        threshold = 0.1

        st.write(f"### {title}")

        # Get voxel indices where value > threshold
        indices = np.argwhere(data > threshold)
        values = data[data > threshold]

        n_total = indices.shape[0]
        n_sample = int(n_total * keep_ratio)

        if n_total == 0:
            st.warning("No voxels above threshold.")
            return

        st.write(f"Displaying {n_sample} points from {n_total} voxels above threshold")

        # Downsample
        sample_idx = np.random.choice(n_total, n_sample, replace=False)
        indices = indices[sample_idx]
        values = values[sample_idx]

        # Convert voxel indices to world coordinates using NIfTI affine
        homogeneous_coords = np.c_[indices, np.ones(n_sample)]
        world_coords = homogeneous_coords @ affine.T
        world_coords = world_coords[:, :3]

        # Create PyVista point cloud
        cloud = pv.PolyData(world_coords)
        cloud["intensity"] = values  # Use voxel value as scalar for coloring

        # Plot
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("black")

        plotter.add_points(cloud, scalars="intensity",
                         cmap="viridis",
                         render_points_as_spheres=True,
                         point_size=5)
        plotter.add_axes()
        plotter.view_isometric()

        # Render in Streamlit
        stpyvista(plotter, key=f"pyvista_{key_suffix}")

    except Exception as e:
        st.error(f"Error visualizing {title.lower()}: {str(e)}")


def display_segmentation_isosurface(segmentation_path, is_full_scan=True):
    """Display a 3D isosurface visualization of segmentation using PyVista."""
    if not os.path.exists(segmentation_path):
        st.error(f"Segmentation file not found: {segmentation_path}")
        return

    try:
        # Load segmentation data
        img = nib.load(segmentation_path)
        data = img.get_fdata()
        affine = img.affine

        # Set title based on full scan or bump level
        if is_full_scan:
            title = "Full-Scan 3D Isosurface"
            key_suffix = "full_scan"
            # Downsample data for faster rendering (only for full scan)
            downsample_factor = 3  # Increased from 2 to 3 for better performance
            data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]
            affine = np.copy(affine)
            affine[:3, :3] *= downsample_factor
        else:
            title = "Bump-Level 3D Isosurface"
            key_suffix = "bump_level"

        st.write(f"### {title}")

        # Create dimensions, origin, and spacing for PyVista grids
        dims = np.array(data.shape)
        # First determine origin and spacing from the affine matrix
        origin = affine[:3, 3]
        spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))

        # Define class-to-color mapping
        color_map = {
            1: "red",
            2: "green",
            3: "blue",
            4: "yellow"
        }

        # Set up plotter with lower detail for faster rendering
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("black")

        # Get unique class values (round to handle potential floating point inaccuracies, include 0)
        data_flat = data.flatten()
        # Consider values >= 0 as potential classes, round them to nearest int
        potential_classes = np.round(data_flat[data_flat > 0.1]).astype(int)
        unique_classes = np.unique(potential_classes)

        # If no classes found, show a message
        if len(unique_classes) == 0:
            st.warning("No classes found in the segmentation data.")
            return

        st.write(f"Found classes: {unique_classes.tolist()}")

        # Track if we successfully added any meshes
        has_meshes = False

        # Process each class separately to avoid the wrapping issue
        for cls in unique_classes:
            # Create a binary mask for just this class
            class_mask = (np.round(data) == cls).astype(float)

            # Create a separate grid for this class
            grid_class = pv.ImageData(
                dimensions=dims,
                origin=origin,
                spacing=spacing
            )
            grid_class.point_data["class"] = class_mask.flatten(order="F")

            # Extract isosurface for just this class
            iso_value = 0.5  # Threshold for binary mask

            # Get color for this class (default to white if not in mapping)
            color = color_map.get(cls, "white")

            try:
                # Extract the isosurface
                isosurface = grid_class.contour([iso_value], scalars="class")

                # Simplify mesh to reduce complexity (adjust parameters for best performance)
                decimated = isosurface.decimate_pro(0.8)  # Reduce mesh to 20% of original size

                # Add the simplified mesh with the specified color and full opacity
                plotter.add_mesh(decimated, color=color, opacity=1.0,
                              label=f"Class {cls}", smooth_shading=True)
                has_meshes = True
            except Exception as e:
                st.warning(f"Could not create isosurface for class {cls}: {str(e)}")

        plotter.add_axes()
        plotter.view_isometric()

        # Only add legend if we have meshes
        if has_meshes:
            plotter.add_legend()

        # Render in Streamlit with a reasonable timeout
        stpyvista(plotter, key=f"isosurface_{key_suffix}")

    except Exception as e:
        st.error(f"Error visualizing {title.lower()} isosurface: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def display_metrology_results(csv_path):
    """Display the metrology analysis results."""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.write("### Metrology Results")
            st.dataframe(df)

            # Create some simple visualizations
            st.write("### Metrology Visualization")
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                ax.hist(df['BLT'], bins=10)
                ax.set_title('Bond Line Thickness Distribution')
                ax.set_xlabel('BLT (μm)')
                ax.set_ylabel('Count')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                ax.hist(df['Void_to_solder_ratio'], bins=10)
                ax.set_title('Void to Solder Ratio Distribution')
                ax.set_xlabel('Ratio')
                ax.set_ylabel('Count')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error displaying metrology results: {str(e)}")
    else:
        st.error(f"Metrology CSV file not found: {csv_path}")


def provide_pdf_download(pdf_path):
    """Provide a download link for the PDF without page refresh."""
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()

        # Get the original filename
        pdf_filename = os.path.basename(pdf_path)

        # Base64 encode the PDF to allow downloading without page refresh
        import base64
        b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

        # Create a custom HTML download link with a generic name to avoid revealing the actual filename
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="analysis_report.pdf" target="_blank"><button style="background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;font-size:16px;">Download Analysis Report (PDF)</button></a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.error(f"PDF report not found: {pdf_path}")


def get_available_samples():
    """Get available samples from output/demo directory."""
    base_path = "output/demo"
    samples = []

    # Check if base path exists
    if not os.path.exists(base_path):
        return []

    # Get all directories in the base path
    samples_level1 = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # For each level 1 directory, get its subdirectories
    for sample_l1 in samples_level1:
        level1_path = os.path.join(base_path, sample_l1)
        samples_level2 = [d for d in os.listdir(level1_path) if os.path.isdir(os.path.join(level1_path, d))]

        # Add each level2 sample with its full path structure
        for sample_l2 in samples_level2:
            sample_path = f"{sample_l1}/{sample_l2}"
            samples.append(sample_path)

    return samples


def display_segmentation_with_2d_slices(segmentation_path, original_data_path, is_full_scan=True, quality_label="", standard_height=None, standard_width=None):
    """Display 2D slices in a single row and 3D rendering below them."""
    # Check if both required files exist
    if not os.path.exists(segmentation_path):
        st.error(f"Segmentation file not found: {segmentation_path}")
        return

    if not os.path.exists(original_data_path):
        st.error(f"Original data file not found: {original_data_path}")
        return

    try:
        # Load original 3D data
        orig_img = nib.load(original_data_path)
        orig_data = orig_img.get_fdata()

        # Set title based on full scan or bump level
        if is_full_scan:
            title = "Full-Scan Segmentation Visualization"
            key_suffix = "full_scan_combined"
            # Downsample data for faster rendering (only for full scan)
            downsample_factor = 3  # Increased from 2 to 3 for better performance
            orig_data = orig_data[::downsample_factor, ::downsample_factor, ::downsample_factor]
        else:
            title = f"Multi-View ({quality_label})"
            key_suffix = f"bump_level_combined_{quality_label.replace(' ', '_').replace('(', '').replace(')', '')}"

        st.write(f"### {title}")

        # Get middle slices for each dimension
        x_mid = orig_data.shape[0] // 2
        y_mid = orig_data.shape[1] // 2
        z_mid = orig_data.shape[2] // 2

        # Normalize data for consistent display
        if orig_data.max() > 0:
            norm_data = orig_data / orig_data.max()
        else:
            norm_data = orig_data

        # Create a row for 2D slices
        col1, col2, col3 = st.columns(3)

        # Calculate consistent size for all slices if not provided
        # Get the shapes of the three slices
        x_slice_shape = norm_data[x_mid, :, :].T.shape
        y_slice_shape = norm_data[:, y_mid, :].T.shape
        z_slice_shape = norm_data[:, :, z_mid].T.shape

        # Find minimum dimensions within this sample unless standard dimensions were provided
        if standard_height is None or standard_width is None:
            min_height = min(x_slice_shape[0], y_slice_shape[0], z_slice_shape[0])
            min_width = min(x_slice_shape[1], y_slice_shape[1], z_slice_shape[1])
        else:
            min_height = standard_height
            min_width = standard_width

        # Create center-cropped slices
        def center_crop_2d(image, target_height, target_width):
            height, width = image.shape
            top = (height - target_height) // 2
            left = (width - target_width) // 2
            # Ensure we don't try to crop beyond image boundaries
            if target_height > height:
                target_height = height
                top = 0
            if target_width > width:
                target_width = width
                left = 0
            return image[top:top+target_height, left:left+target_width]

        x_slice = center_crop_2d(norm_data[x_mid, :, :].T, min_height, min_width)
        y_slice = center_crop_2d(norm_data[:, y_mid, :].T, min_height, min_width)
        z_slice = center_crop_2d(norm_data[:, :, z_mid].T, min_height, min_width)

        # Display 2D slices in a row
        with col1:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.imshow(x_slice, cmap='gray')
            ax1.set_title(f"X Slice (Mid={x_mid})")
            ax1.axis('off')
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(y_slice, cmap='gray')
            ax2.set_title(f"Y Slice (Mid={y_mid})")
            ax2.axis('off')
            st.pyplot(fig2)

        with col3:
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            ax3.imshow(z_slice, cmap='gray')
            ax3.set_title(f"Z Slice (Mid={z_mid})")
            ax3.axis('off')
            st.pyplot(fig3)

        # Create a section for the 3D visualization below the 2D slices
        st.write(f"### 3D Visualization")

        # Load segmentation data for 3D viz
        img = nib.load(segmentation_path)
        data = img.get_fdata()
        affine = img.affine

        # Downsample for faster rendering
        if is_full_scan:
            downsample_factor = 3  # Increased from 2 to 3 for better performance
            data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]
            affine = np.copy(affine)
            affine[:3, :3] *= downsample_factor

        # Create a PyVista ImageData object
        dims = np.array(data.shape)
        origin = affine[:3, 3]
        spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))

        # Define class-to-color mapping
        color_map = {
            1: "red", 2: "green", 3: "blue", 4: "yellow"
        }

        # Set up plotter with lower detail for faster rendering
        plotter = pv.Plotter(off_screen=True, window_size=[500, 600])  # Further reduced window size
        plotter.set_background("black")

        # Process segmentation data
        data_flat = data.flatten()
        # Consider values >= 0 as potential classes, round them to nearest int
        potential_classes = np.round(data_flat[data_flat > 0.1]).astype(int)
        unique_classes = np.unique(potential_classes)

        if len(unique_classes) == 0:
            st.warning("No classes found in the segmentation data.")
            return

        # Process each class separately to avoid the wrapping issue
        has_meshes = False
        for cls in unique_classes:
            # Create a binary mask for just this class
            class_mask = (np.round(data) == cls).astype(float)

            # Add the mask to a new grid for this class
            grid_class = pv.ImageData(
                dimensions=dims,
                origin=origin,
                spacing=spacing
            )
            grid_class.point_data["class"] = class_mask.flatten(order="F")

            # Extract isosurface for just this class
            iso_value = 0.5  # Threshold for binary mask
            color = color_map.get(cls, "white")

            try:
                # Extract isosurface
                isosurface = grid_class.contour([iso_value], scalars="class")

                # Simplify mesh
                decimated = isosurface.decimate_pro(0.5)  # Reduce mesh to 50% of original

                plotter.add_mesh(decimated, color=color, opacity=0.95,
                                label=f"Class {cls}", smooth_shading=True)
                has_meshes = True
            except Exception as e:
                st.warning(f"Could not create isosurface for class {cls}: {str(e)}")

        plotter.add_axes()
        plotter.view_isometric()

        if has_meshes:
            plotter.add_legend()

        # Render in Streamlit with a reasonable timeout
        stpyvista(plotter, key=f"isosurface_{key_suffix}")

        # Return the calculated dimensions for consistency across samples
        return min_height, min_width

    except Exception as e:
        st.error(f"Error creating multi-view visualization: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def simulate_pipeline(selected_sample):
    # Define the output path based on the selected sample
    sample_path = os.path.join("output/demo", selected_sample)

    # Load bb2d.yaml configuration
    bb2d_config = load_bb2d_config(selected_sample)

    # Calculate common crop dimensions for both views
    target_width, target_height = calculate_crop_dimensions(sample_path, bb2d_config)

    # Function to create a phase container with consistent styling
    def create_phase_container(title):
        container = st.container()
        with container:
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:20px; background-color:#f9f9f9">
                <h3 style="margin-top:0">{title}</h3>
                <div id="content"></div>
            </div>
            """, unsafe_allow_html=True)
        return container

    # Create containers for each phase
    conversion_container = create_phase_container("Phase 1: NII to JPG Conversion")
    detection_container = create_phase_container("Phase 2: Object Detection")
    segmentation_container = create_phase_container("Phase 3: 3D Segmentation")
    metrology_container = create_phase_container("Phase 4: Metrology Analysis")

    try:
        # Step 1: Convert NII to JPG (Simulation)
        with conversion_container:
            status = st.empty()
            progress = st.empty()
            result = st.container()

            status.text("Converting 3D NII to 2D images...")
            with timer("NII to JPG Conversion"):
                images_path_view1, images_path_view2 = simulate_conversion(
                    sample_path,
                    lambda p: progress.progress(p)
                )

            # Keep progress bar at 100% after completion
            progress.progress(1.0)
            status.success("Conversion complete!")

            # Display sample images from both views using bb2d.yaml configuration
            with result:
                col1, col2 = st.columns(2)

                # Display View 1
                with col1:
                    st.write("View 1")
                    if bb2d_config["view1"]:
                        # Use the specific image from config
                        image_path = os.path.join(images_path_view1, bb2d_config["view1"])
                        display_image_file(image_path, target_width, target_height)
                    else:
                        # Fallback to displaying a middle slice if config doesn't specify
                        image_files_view1 = sorted(glob.glob(os.path.join(images_path_view1, "*.jpg")))
                        if image_files_view1:
                            middle_image_view1 = image_files_view1[len(image_files_view1) // 2]
                            display_image_file(middle_image_view1, target_width, target_height)
                        else:
                            st.warning("No images found for View 1")

                # Display View 2
                with col2:
                    st.write("View 2")
                    if bb2d_config["view2"]:
                        # Use the specific image from config
                        image_path = os.path.join(images_path_view2, bb2d_config["view2"])
                        display_image_file(image_path, target_width, target_height)
                    else:
                        # Fallback to displaying a middle slice if config doesn't specify
                        image_files_view2 = sorted(glob.glob(os.path.join(images_path_view2, "*.jpg")))
                        if image_files_view2:
                            middle_image_view2 = image_files_view2[len(image_files_view2) // 2]
                            display_image_file(middle_image_view2, target_width, target_height)
                        else:
                            st.warning("No images found for View 2")

        # Step 2: Object Detection (Simulation)
        with detection_container:
            status = st.empty()
            progress = st.empty()
            result = st.container()

            status.text("Running object detection...")
            with timer("Object Detection"):
                detection_path_view1, detection_path_view2 = simulate_object_detection(
                    sample_path,
                    lambda p: progress.progress(p)
                )

            # Keep progress bar at 100% after completion
            progress.progress(1.0)
            status.success("Object detection complete!")

            # Display detection results using bb2d.yaml configuration
            with result:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("View 1")
                    if bb2d_config["view1"]:
                        # Use the specific image from config
                        image_path = os.path.join(detection_path_view1, bb2d_config["view1"])
                        display_image_file(image_path, target_width, target_height)
                    else:
                        # Fallback to showing the largest file
                        image_files = glob.glob(os.path.join(detection_path_view1, "*.jpg"))
                        if image_files:
                            image_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                            display_image_file(image_files[0], target_width, target_height)
                        else:
                            st.warning("No detection visualization images found for View 1")

                with col2:
                    st.write("View 2")
                    if bb2d_config["view2"]:
                        # Use the specific image from config
                        image_path = os.path.join(detection_path_view2, bb2d_config["view2"])
                        display_image_file(image_path, target_width, target_height)
                    else:
                        # Fallback to showing the largest file
                        image_files = glob.glob(os.path.join(detection_path_view2, "*.jpg"))
                        if image_files:
                            image_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                            display_image_file(image_files[0], target_width, target_height)
                        else:
                            st.warning("No detection visualization images found for View 2")

        # Run the 3D bounding box generation (no visible phase)
        with timer("3D Bounding Box Generation"):
            try:
                bb3d_path = simulate_3d_bbox_generation(sample_path, None)
            except FileNotFoundError as e:
                st.error(f"Error: {str(e)}")
                bb3d_path = None

        # Step 3: 3D Segmentation (Simulation)
        with segmentation_container:
            status = st.empty()
            progress = st.empty()
            result = st.container()

            status.text("Performing 3D segmentation...")
            try:
                with timer("3D Segmentation"):
                    good_seg_path, bad_seg_path = simulate_3d_segmentation(
                        sample_path,
                        bb2d_config,
                        lambda p: progress.progress(p)
                    )

                # Keep progress bar at 100% after completion
                progress.progress(1.0)
                status.success("3D segmentation complete!")

                # Display segmentation results
                with result:
                    # First display bounding box information (moved from previous phase)
                    if bb3d_path and os.path.exists(bb3d_path):
                        try:
                            bb_data = np.load(bb3d_path)
                            st.write(f"Generated {bb_data.shape[1]} 3D bounding boxes")
                        except Exception as e:
                            st.error(f"Error loading 3D bounding box data: {str(e)}")

                    # Path to full-scan segmentation
                    full_scan_path = os.path.join(sample_path, "class_0_segmentation.nii.gz")

                    # Determine original data paths based on bb2d config
                    good_seg_file = bb2d_config.get("good")
                    bad_seg_file = bb2d_config.get("bad")

                    if good_seg_file:
                        original_good_data_path = os.path.join(sample_path, "mmt", "img", "class_0", good_seg_file.replace("pred", "img"))
                    else:
                        original_good_data_path = None
                        st.warning("'good' segmentation file not specified in bb2d.yaml, cannot determine original data path.")

                    if bad_seg_file:
                        original_bad_data_path = os.path.join(sample_path, "mmt", "img", "class_0", bad_seg_file.replace("pred", "img"))
                    else:
                        original_bad_data_path = None
                        st.warning("'bad' segmentation file not specified in bb2d.yaml, cannot determine original data path.")

                    # Display full scan segmentation if file exists
                    if os.path.exists(full_scan_path):
                        st.write("## Full Scan Visualization")
                        st.write("Showing global view of the entire scan volume.")

                        display_segmentation_isosurface(full_scan_path, is_full_scan=True)

                    # Add a separator between visualizations
                    st.markdown("---")
                    st.write("## Bump-Level Visualization")

                    # Pre-calculate common dimensions for both good and bad bump visualizations
                    standard_height = None
                    standard_width = None

                    # If both paths exist, load them to calculate common dimensions
                    if original_good_data_path and os.path.exists(original_good_data_path) and \
                       original_bad_data_path and os.path.exists(original_bad_data_path):
                        try:
                            # Load good bump data for size calculation
                            good_img = nib.load(original_good_data_path)
                            good_data = good_img.get_fdata()

                            # Get middle slices for each dimension of good bump
                            good_x_mid = good_data.shape[0] // 2
                            good_y_mid = good_data.shape[1] // 2
                            good_z_mid = good_data.shape[2] // 2

                            # Get slice shapes for good bump
                            good_x_slice_shape = good_data[good_x_mid, :, :].T.shape
                            good_y_slice_shape = good_data[:, good_y_mid, :].T.shape
                            good_z_slice_shape = good_data[:, :, good_z_mid].T.shape

                            # Load bad bump data for size calculation
                            bad_img = nib.load(original_bad_data_path)
                            bad_data = bad_img.get_fdata()

                            # Get middle slices for each dimension of bad bump
                            bad_x_mid = bad_data.shape[0] // 2
                            bad_y_mid = bad_data.shape[1] // 2
                            bad_z_mid = bad_data.shape[2] // 2

                            # Get slice shapes for bad bump
                            bad_x_slice_shape = bad_data[bad_x_mid, :, :].T.shape
                            bad_y_slice_shape = bad_data[:, bad_y_mid, :].T.shape
                            bad_z_slice_shape = bad_data[:, :, bad_z_mid].T.shape

                            # Calculate minimum dimensions across both samples
                            standard_height = min(
                                good_x_slice_shape[0], good_y_slice_shape[0], good_z_slice_shape[0],
                                bad_x_slice_shape[0], bad_y_slice_shape[0], bad_z_slice_shape[0]
                            )
                            standard_width = min(
                                good_x_slice_shape[1], good_y_slice_shape[1], good_z_slice_shape[1],
                                bad_x_slice_shape[1], bad_y_slice_shape[1], bad_z_slice_shape[1]
                            )

                            st.info(f"Using standardized dimensions for bump visualizations: {standard_width}x{standard_height} pixels")
                        except Exception as e:
                            st.warning(f"Error calculating common dimensions: {str(e)}. Will use individual sample dimensions.")

                    # Create two columns for good and bad visualizations
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("### Good Quality Bump")
                        if original_good_data_path and os.path.exists(original_good_data_path):
                            # Get bump index from the filename in bb2d_config
                            good_bump_label = "Good"
                            if bb2d_config.get("good"):
                                good_filename = os.path.basename(bb2d_config.get("good"))
                                # Try to extract bump index if it follows naming convention
                                if good_filename.startswith("pred_") and good_filename.endswith(".nii.gz"):
                                    try:
                                        bump_id = good_filename.replace("pred_", "").replace(".nii.gz", "")
                                        good_bump_label = f"Good (Bump {bump_id})"
                                    except:
                                        pass
                            display_segmentation_with_2d_slices(good_seg_path, original_good_data_path,
                                                               is_full_scan=False, quality_label=good_bump_label,
                                                               standard_height=standard_height,
                                                               standard_width=standard_width)
                        elif original_good_data_path:
                            st.error(f"Original data file not found: {original_good_data_path}")
                        else:
                             st.error("Cannot display good bump visualization without original data path.")

                    with col2:
                        st.write("### Bad Quality Bump")
                        if original_bad_data_path and os.path.exists(original_bad_data_path):
                            # Get bump index from the filename in bb2d_config
                            bad_bump_label = "Bad"
                            if bb2d_config.get("bad"):
                                bad_filename = os.path.basename(bb2d_config.get("bad"))
                                # Try to extract bump index if it follows naming convention
                                if bad_filename.startswith("pred_") and bad_filename.endswith(".nii.gz"):
                                    try:
                                        bump_id = bad_filename.replace("pred_", "").replace(".nii.gz", "")
                                        bad_bump_label = f"Bad (Bump {bump_id})"
                                    except:
                                        pass
                            display_segmentation_with_2d_slices(bad_seg_path, original_bad_data_path,
                                                               is_full_scan=False, quality_label=bad_bump_label,
                                                               standard_height=standard_height,
                                                               standard_width=standard_width)
                        elif original_bad_data_path:
                            st.error(f"Original data file not found: {original_bad_data_path}")
                        else:
                            st.error("Cannot display bad bump visualization without original data path.")
            except FileNotFoundError as e:
                # Failed to find segmentation files
                progress.progress(1.0)
                status.error(f"Segmentation failed: {str(e)}")

        # Step 4: Metrology and Report Generation (Simulation)
        with metrology_container:
            status = st.empty()
            progress = st.empty()
            result = st.container()

            status.text("Running metrology analysis and generating report...")
            try:
                with timer("Metrology and Report"):
                    csv_path, pdf_path = simulate_metrology(
                        sample_path,
                        lambda p: progress.progress(p)
                    )

                # Keep progress bar at 100% after completion
                progress.progress(1.0)
                status.success("Metrology analysis complete!")

                # Display metrology results and provide report download
                with result:
                    display_metrology_results(csv_path)

                    # Use the no-refresh PDF download
                    provide_pdf_download(pdf_path)
            except FileNotFoundError as e:
                # Failed to find metrology files
                progress.progress(1.0)
                status.error(f"Metrology analysis failed: {str(e)}")

        st.success("Processing complete!")
        return True

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def process_uploaded_nifti(file_path):
    """Process an uploaded NIfTI file through the pipeline."""
    # Create a temporary output directory for this file
    base_name = os.path.basename(file_path).split('.')[0]
    temp_output_dir = os.path.join("temp_uploads", f"output_{base_name}_{int(time.time())}")
    os.makedirs(temp_output_dir, exist_ok=True)

    # Function to create a phase container with consistent styling
    def create_phase_container(title):
        container = st.container()
        with container:
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:20px; background-color:#f9f9f9">
                <h3 style="margin-top:0">{title}</h3>
                <div id="content"></div>
            </div>
            """, unsafe_allow_html=True)
        return container

    # Create containers for each phase
    conversion_container = create_phase_container("Phase 1: NII to 3D Visualization")
    segmentation_container = create_phase_container("Phase 2: 3D Segmentation (Placeholder)")
    metrology_container = create_phase_container("Phase 3: Metrology Analysis (Placeholder)")

    try:
        # Phase 1: NII Visualization
        with conversion_container:
            status = st.empty()
            progress = st.empty()
            result = st.container()

            status.text("Loading NIfTI data...")

            with timer("NIfTI Loading and Visualization"):
                try:
                    # Load NIfTI data
                    nii_img = nib.load(file_path)
                    nii_data = nii_img.get_fdata()

                    # Update progress with configurable delays
                    steps = SIMULATION_DELAYS["progress_steps"]
                    step_delay = SIMULATION_DELAYS["progress_delay"]
                    for i in range(steps):
                        time.sleep(step_delay)
                        progress.progress((i+1)/steps)

                    # Display basic info
                    st.write(f"Data shape: {nii_data.shape}")
                    st.write(f"Data type: {nii_data.dtype}")

                    # Create a simple visualization - multi-slice viewer for 3D data
                    with result:
                        st.write("### NIfTI Data Visualization")

                        # Get the middle slices for each dimension
                        x_mid = nii_data.shape[0] // 2
                        y_mid = nii_data.shape[1] // 2
                        z_mid = nii_data.shape[2] // 2

                        # Create sliders for browsing through slices
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            x_slice = st.slider("X Slice", 0, nii_data.shape[0]-1, x_mid)
                            fig, ax = plt.subplots()
                            ax.imshow(nii_data[x_slice, :, :].T, cmap='gray')
                            ax.set_title(f"X Slice: {x_slice}")
                            st.pyplot(fig)

                        with col2:
                            y_slice = st.slider("Y Slice", 0, nii_data.shape[1]-1, y_mid)
                            fig, ax = plt.subplots()
                            ax.imshow(nii_data[:, y_slice, :].T, cmap='gray')
                            ax.set_title(f"Y Slice: {y_slice}")
                            st.pyplot(fig)

                        with col3:
                            z_slice = st.slider("Z Slice", 0, nii_data.shape[2]-1, z_mid)
                            fig, ax = plt.subplots()
                            ax.imshow(nii_data[:, :, z_slice].T, cmap='gray')
                            ax.set_title(f"Z Slice: {z_slice}")
                            st.pyplot(fig)

                        # 3D visualization with PyVista
                        st.write("### 3D Volume Rendering")

                        # Create a PyVista grid from the numpy array
                        grid = pv.UniformGrid()
                        grid.dimensions = np.array(nii_data.shape) + 1
                        grid.origin = (0, 0, 0)  # assuming (0,0,0) origin
                        grid.spacing = (1, 1, 1)  # assuming unit spacing

                        # Fill the grid with the numpy array
                        grid.cell_data["values"] = nii_data.flatten(order="F")

                        # Set up the plotter
                        plotter = pv.Plotter(off_screen=True)
                        plotter.add_volume(grid, cmap="viridis", opacity="linear")
                        plotter.view_isometric()

                        # Render in Streamlit
                        stpyvista(plotter, key="pyvista_volume")

                except Exception as e:
                    st.error(f"Error processing NIfTI file: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    return False

            # Keep progress bar at 100% after completion
            progress.progress(1.0)
            status.success("NIfTI visualization complete!")

        # Phase 2: Segmentation (placeholder)
        with segmentation_container:
            st.info("This is a placeholder for the segmentation phase. In a full implementation, this would perform segmentation on the uploaded NIfTI data.")
            st.warning("Segmentation processing skipped for uploaded files.")

        # Phase 3: Metrology (placeholder)
        with metrology_container:
            st.info("This is a placeholder for the metrology phase. In a full implementation, this would perform measurements on the segmented data.")
            st.warning("Metrology processing skipped for uploaded files.")

        st.success("Processing complete!")
        return True

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def main():
    title = "3D-IntelliScan"
    st.set_page_config(page_title=title, page_icon="icon.png", layout="wide")

    # Display logo in the header
    st.image("logo.png", width=200)

    st.title(title)
    st.markdown("### AI Solution for 3D Metrology and Defect Detection for Semiconductor Manufacturing")

    # Track which data source is selected
    selected_source = st.radio("Select data source:", ["Server Sample", "Upload Your Own Data (In Development)"], horizontal=True)
    st.write("---")

    # Variables to track selected data
    selected_sample = None
    temp_file_path = None

    if selected_source == "Server Sample":
        # Get available samples
        available_samples = get_available_samples()

        if not available_samples:
            st.error("No samples found in the output/demo directory. Please check the server configuration.")
        else:
            # Add blank option as default
            sample_options = [""] + available_samples
            selected_sample = st.selectbox(
                "Select a sample to process",
                options=sample_options,
                index=0
            )

            if selected_sample:
                st.success(f"Selected sample: {selected_sample}")

    else:  # Upload Your Own Data
        st.write("Upload your own NIfTI (.nii or .nii.gz) file for processing.")

        # Remove the file size restriction for uploads
        uploaded_file = st.file_uploader(
            "Upload NIfTI file",
            type=["nii", "nii.gz"],
            accept_multiple_files=False,
            help="Upload a NIfTI file to process through the pipeline."
        )

        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")
            # Display file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"File size: {file_size_mb:.2f} MB")

            # Save the uploaded file to a temporary location
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write(f"File saved to: {temp_file_path}")

            # Option to view the NIfTI header information
            if st.checkbox("View NIfTI Header Info"):
                try:
                    nii_img = nib.load(temp_file_path)
                    header_info = {
                        "Data shape": nii_img.shape,
                        "Data type": nii_img.get_data_dtype(),
                        "Affine transform": nii_img.affine.tolist(),
                        "Header info": {k: v for k, v in nii_img.header.items()}
                    }
                    st.json(header_info)
                except Exception as e:
                    st.error(f"Error reading NIfTI header: {str(e)}")

    # Process button - only enable if a sample is selected or a file is uploaded
    process_data = selected_sample if selected_sample else temp_file_path
    process_ready = process_data is not None

    if st.button("Start Processing", disabled=not process_ready):
        if not process_ready:
            st.error("Please select a sample or upload a file before processing.")
        else:
            with st.spinner("Processing..."):
                st.write("Starting pipeline processing...")

                if selected_sample:
                    # Process server sample
                    simulate_pipeline(selected_sample)
                else:
                    # Process uploaded file
                    process_uploaded_nifti(temp_file_path)


if __name__ == "__main__":
    main()