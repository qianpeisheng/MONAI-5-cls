#!/usr/bin/env python3
"""
Test script to verify supervoxel label override functionality.

Tests:
1. Loading .npy files with MONAI transforms
2. override_train_labels() function logic
3. Shape and value compatibility between .npy SV labels and .nii GT labels
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path to import from train script
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_finetune_wp5 import override_train_labels

# Test MONAI transforms with .npy files
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd

def test_npy_loading():
    """Test that MONAI LoadImaged can load .npy supervoxel labels."""
    print("\n" + "="*60)
    print("TEST 1: Loading .npy files with MONAI transforms")
    print("="*60)

    # Use a known SV label file
    sv_dir = Path("/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n20000_c0.05_s1.0_ras2_voted")

    # Find first available case
    npy_files = list(sv_dir.glob("*_labels.npy"))
    if not npy_files:
        print("❌ FAILED: No .npy label files found in", sv_dir)
        return False

    test_file = npy_files[0]
    print(f"Testing with: {test_file.name}")

    # Create transform pipeline
    transforms = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
    ])

    # Test loading
    try:
        data = {"label": str(test_file)}
        result = transforms(data)

        label_tensor = result["label"]
        print(f"✓ Successfully loaded .npy file")
        print(f"  - Type: {type(label_tensor)}")
        print(f"  - Shape: {label_tensor.shape}")
        print(f"  - Dtype: {label_tensor.dtype}")
        print(f"  - Value range: [{label_tensor.min():.1f}, {label_tensor.max():.1f}]")
        print(f"  - Unique values: {torch.unique(label_tensor).numpy()}")

        # Validate
        assert isinstance(label_tensor, torch.Tensor), "Should be torch.Tensor"
        assert label_tensor.ndim == 4, "Should be 4D (C,X,Y,Z)"
        assert label_tensor.shape[0] == 1, "Should have 1 channel"
        assert label_tensor.min() >= 0, "Values should be >= 0"
        assert label_tensor.max() <= 4, "Values should be <= 4"

        print("✓ PASSED: .npy files load correctly with MONAI transforms\n")
        return True

    except Exception as e:
        print(f"❌ FAILED: Error loading .npy file: {e}\n")
        return False


def test_override_function():
    """Test the override_train_labels function."""
    print("\n" + "="*60)
    print("TEST 2: override_train_labels() function")
    print("="*60)

    # Load actual datalist
    datalist_file = Path("datalist_train.json")
    if not datalist_file.exists():
        print("❌ FAILED: datalist_train.json not found")
        return False

    with open(datalist_file) as f:
        train_list = json.load(f)

    # Take first 5 cases for testing
    test_list = train_list[:5]
    print(f"Testing with {len(test_list)} cases from datalist")

    # Test override directory
    override_dir = Path("/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n20000_c0.05_s1.0_ras2_voted")

    try:
        # Run override function
        result_list = override_train_labels(test_list, override_dir)

        print(f"✓ Override function completed successfully")
        print(f"  - Input cases: {len(test_list)}")
        print(f"  - Output cases: {len(result_list)}")

        # Validate results
        assert len(result_list) == len(test_list), "Should have same number of cases"

        for i, (orig, overridden) in enumerate(zip(test_list, result_list)):
            print(f"\nCase {i+1}: {orig['id']}")
            print(f"  Original label:   {Path(orig['label']).name}")
            print(f"  Overridden label: {Path(overridden['label']).name}")

            # Verify image path unchanged
            assert orig['image'] == overridden['image'], "Image path should be unchanged"

            # Verify label path changed
            assert orig['label'] != overridden['label'], "Label path should be changed"

            # Verify new label file exists
            assert Path(overridden['label']).exists(), f"Override label should exist: {overridden['label']}"

            # Verify naming pattern
            override_path = Path(overridden['label'])
            assert override_path.name.endswith('_labels.npy'), "Should use .npy SV label format"
            assert orig['id'] in override_path.name, "Should contain case ID"

        print("\n✓ PASSED: All cases correctly mapped to override labels\n")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_shape_compatibility():
    """Test that .npy SV labels have same shape as .nii GT labels."""
    print("\n" + "="*60)
    print("TEST 3: Shape compatibility between SV and GT labels")
    print("="*60)

    # Load datalist
    datalist_file = Path("datalist_train.json")
    with open(datalist_file) as f:
        train_list = json.load(f)

    # Test first case
    case = train_list[0]
    case_id = case['id']
    gt_label_path = Path(case['label'])
    sv_label_path = Path("/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n20000_c0.05_s1.0_ras2_voted") / f"{case_id}_labels.npy"

    print(f"Testing case: {case_id}")
    print(f"  GT label: {gt_label_path.name}")
    print(f"  SV label: {sv_label_path.name}")

    # Load both with MONAI transforms
    transforms = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
    ])

    try:
        # Load GT
        gt_data = transforms({"label": str(gt_label_path)})
        gt_label = gt_data["label"]

        # Load SV
        sv_data = transforms({"label": str(sv_label_path)})
        sv_label = sv_data["label"]

        print(f"\nGT label (from .nii):")
        print(f"  - Shape: {gt_label.shape}")
        print(f"  - Dtype: {gt_label.dtype}")
        print(f"  - Value range: [{gt_label.min():.1f}, {gt_label.max():.1f}]")

        print(f"\nSV label (from .npy):")
        print(f"  - Shape: {sv_label.shape}")
        print(f"  - Dtype: {sv_label.dtype}")
        print(f"  - Value range: [{sv_label.min():.1f}, {sv_label.max():.1f}]")

        # Compare
        assert gt_label.shape == sv_label.shape, f"Shapes should match: {gt_label.shape} vs {sv_label.shape}"

        # Check Dice overlap (should be high but not perfect due to SV approximation)
        intersection = ((gt_label == sv_label).sum().item())
        union = gt_label.numel()
        agreement = intersection / union

        print(f"\nVoxel-wise agreement: {agreement:.4f} ({agreement*100:.2f}%)")

        assert agreement > 0.8, f"Agreement should be high (>80%), got {agreement:.2%}"

        print("✓ PASSED: SV and GT labels are compatible\n")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_missing_files_error():
    """Test that override function raises error for missing files."""
    print("\n" + "="*60)
    print("TEST 4: Error handling for missing override files")
    print("="*60)

    # Create fake datalist with non-existent case
    fake_list = [{
        "id": "NONEXISTENT_CASE_12345",
        "image": "/fake/path/image.nii",
        "label": "/fake/path/label.nii"
    }]

    override_dir = Path("/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n20000_c0.05_s1.0_ras2_voted")

    try:
        result = override_train_labels(fake_list, override_dir)
        print("❌ FAILED: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"✓ Correctly raised FileNotFoundError:")
        print(f"  {str(e)[:200]}...")
        print("✓ PASSED: Missing files are properly detected\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SUPERVOXEL LABEL OVERRIDE - TEST SUITE")
    print("="*70)
    print("\nTesting changes to train_finetune_wp5.py:")
    print("  - override_train_labels() function")
    print("  - Support for .npy label files")
    print("  - Shape/value compatibility")

    results = []

    # Run tests
    results.append(("MONAI .npy loading", test_npy_loading()))
    results.append(("Override function", test_override_function()))
    results.append(("Shape compatibility", test_shape_compatibility()))
    results.append(("Error handling", test_missing_files_error()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("\n" + "="*70)
    print(f"OVERALL: {passed}/{total} tests passed")
    print("="*70 + "\n")

    if passed == total:
        print("🎉 All tests passed! Safe to use --train_label_override_dir flag.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review before using.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
