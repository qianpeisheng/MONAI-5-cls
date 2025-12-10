import os
import inspect
import tempfile
import importlib
import importlib.util
import unittest

_torch_available = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(_torch_available, "torch not available")
class TestEvalSemantics(unittest.TestCase):
    def setUp(self):
        # Fresh import of trainer to ensure latest code
        if 'train_finetune_wp5' in globals():
            importlib.reload(globals()['train_finetune_wp5'])
        else:
            pass

    def test_compute_metrics_both_empty_counts_as_one_and_ignore6(self):
        import torch
        import train_finetune_wp5 as tfw

        # Build small tensors (B=3, 1x4x4x4)
        B = 3
        shape = (B, 1, 4, 4, 4)
        pred = torch.zeros(shape, dtype=torch.int64)
        gt = torch.zeros(shape, dtype=torch.int64)

        # Case 0: both empty for class 1
        # Case 1: present class 1, perfect match
        pred[1, 0, 0, 0, 0] = 1
        gt[1, 0, 0, 0, 0] = 1
        # Case 2: some voxels are label 6, which should be ignored
        gt[2, 0, 1, 1, 1] = 6
        pred[2, 0, 1, 1, 1] = 1  # should be ignored by mask

        res = tfw.compute_metrics(pred, gt, heavy=False)
        # Background class 0: both-empty areas treated normally; not asserting here.
        # Foreground class 1:
        c1 = res[1]
        # Avg across 3 cases: case0 (both empty => 1.0), case1 (perfect => 1.0), case2 (ignored voxel, effectively empty => counts as 1.0)
        self.assertAlmostEqual(c1['dice'], 1.0, places=6)
        self.assertAlmostEqual(c1['iou'], 1.0, places=6)

    def test_evaluate_signature_has_no_empty_pair_policy(self):
        import train_finetune_wp5 as tfw
        sig = inspect.signature(tfw.evaluate)
        self.assertNotIn('empty_pair_policy', sig.parameters)


@unittest.skipUnless(_torch_available, "torch not available")
class TestArgparseAndLogging(unittest.TestCase):
    def test_parse_args_disallows_infer_mode(self):
        import train_finetune_wp5 as tfw
        # Simulate CLI with --mode infer; argparse should error
        with self.assertRaises(SystemExit):
            _ = tfw.parse_args(["--mode", "infer"])  # type: ignore[arg-type]

    def test_logging_to_file_creates_log(self):
        import train_finetune_wp5 as tfw
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "train.log")
            # Use tee-based stdout capture implemented in the trainer
            tfw._init_run_logging(out_dir=td, enable=True, filename=os.path.basename(log_path))
            print("hello-wp5-log")
            # Flush by closing and reopening stdout is not feasible here; rely on write-through
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("hello-wp5-log", content)


class TestSourceConsistency(unittest.TestCase):
    def test_no_duplicate_function_defs(self):
        # Ensure there is only a single definition for critical helpers
        here = os.path.dirname(os.path.dirname(__file__))
        train_py = os.path.join(here, "train_finetune_wp5.py")
        with open(train_py, "r", encoding="utf-8") as f:
            src = f.read()

        def_count = lambda name: src.count(f"def {name}(")
        self.assertEqual(def_count("compute_metrics"), 1)
        self.assertEqual(def_count("build_slice_supervision_mask"), 1)
        self.assertEqual(def_count("build_points_supervision_mask"), 1)


class TestBuildDatalistsBumpDatasetNoTypeFilter(unittest.TestCase):
    def test_build_datalists_does_not_use_type_field(self):
        """
        Ensure that build_datalists() does not call BumpDataset.filter(type=\"...\")
        and instead splits directly on the full dataset.
        """
        import json
        from pathlib import Path
        import types
        import importlib.machinery as machinery
        import train_finetune_wp5 as tfw

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "images").mkdir()
            (data_root / "labels").mkdir()
            (data_root / "metadata.jsonl").write_text("", encoding="utf-8")

            cfg_path = data_root / "dataset_config.json"
            cfg = {
                "loader_version": "1.1.0",
                "test_serial_numbers": [1, 2],
                "train_serial_numbers": [],
            }
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

            loader_py = root / "dataset_loader.py"
            loader_py.write_text("# dummy loader for tests\n", encoding="utf-8")

            class FakeView:
                def __init__(self, parent, indices):
                    self.parent = parent
                    self.indices = indices

                def __len__(self):
                    return len(self.indices)

                def get_metadata(self, i):
                    return self.parent._meta[self.indices[i]]

            class FakeBumpDataset:
                def __init__(self, data_dir: str):
                    # Three samples, with serials 1, 2, 3
                    self.serial_numbers = [1, 2, 3]
                    self._meta = [
                        {"image_path": "img1", "label_path": "lbl1", "pair_id": "case1", "serial": 1},
                        {"image_path": "img2", "label_path": "lbl2", "pair_id": "case2", "serial": 2},
                        {"image_path": "img3", "label_path": "lbl3", "pair_id": "case3", "serial": 3},
                    ]

                def filter(self, **kwargs):
                    raise AssertionError("BumpDataset.filter(type=...) must not be used by build_datalists")

                def split(self, train_serial_numbers=None, test_serial_numbers=None):
                    if train_serial_numbers:
                        train_serials = set(int(s) for s in train_serial_numbers)
                        test_serials = set(int(s) for s in self.serial_numbers if s not in train_serials)
                    elif test_serial_numbers:
                        test_serials = set(int(s) for s in test_serial_numbers)
                        train_serials = set(int(s) for s in self.serial_numbers if s not in test_serials)
                    else:
                        train_serials = set(self.serial_numbers)
                        test_serials = set()

                    train_idx = [i for i, m in enumerate(self._meta) if m["serial"] in train_serials]
                    test_idx = [i for i, m in enumerate(self._meta) if m["serial"] in test_serials]
                    return FakeView(self, train_idx), FakeView(self, test_idx)

            class FakeLoader:
                def __init__(self, name, path):
                    self.name = name
                    self.path = path

                def load_module(self, name):
                    return types.SimpleNamespace(BumpDataset=FakeBumpDataset)

            orig_sfl = machinery.SourceFileLoader
            machinery.SourceFileLoader = FakeLoader  # type: ignore[assignment]
            try:
                train_list, test_list = tfw.build_datalists(data_root, cfg_path)
            finally:
                machinery.SourceFileLoader = orig_sfl  # type: ignore[assignment]

            # With test_serial_numbers [1,2], serials 1 and 2 should be in test, 3 in train.
            self.assertEqual({r["id"] for r in test_list}, {"case1", "case2"})
            self.assertEqual({r["id"] for r in train_list}, {"case3"})


@unittest.skipUnless(_torch_available, "torch not available")
class TestEvalScriptCLI(unittest.TestCase):
    def test_eval_cli_no_empty_pair_policy_and_has_logging(self):
        import scripts.eval_wp5 as ev
        # The script should expose get_parser for testing
        parser = ev.get_parser()
        opts = {a.dest for a in parser._actions}
        self.assertNotIn("empty_pair_policy", opts)
        self.assertIn("log_to_file", opts)
        self.assertIn("log_file_name", opts)

    def test_eval_cli_heavy_default_and_fast_flag(self):
        import scripts.eval_wp5 as ev
        parser = ev.get_parser()
        # Minimal required args for parse_args
        args_default = parser.parse_args(["--ckpt", "x.ckpt", "--output_dir", "outdir"])
        self.assertTrue(getattr(args_default, "heavy", False))

        args_fast = parser.parse_args(["--ckpt", "x.ckpt", "--output_dir", "outdir", "--fast"])
        self.assertFalse(getattr(args_fast, "heavy", True))


@unittest.skipUnless(_torch_available, "torch not available")
class TestEvalLoadTestList(unittest.TestCase):
    def test_load_test_list_prefers_split_cfg_when_provided(self):
        import types
        import scripts.eval_wp5 as ev

        # Stub args with both datalist and data_root/split_cfg
        args = types.SimpleNamespace(
            datalist="datalist_test.json",
            data_root="/tmp/data_root_should_be_preferred",
            split_cfg="/tmp/split_cfg_should_be_preferred",
        )

        # Patch build_datalists to verify it is called and its return used
        def fake_build_datalists(data_dir, cfg_path):
            # data_dir/cfg_path must come from args.data_root/split_cfg
            self.assertIn("data_root_should_be_preferred", str(data_dir))
            self.assertIn("split_cfg_should_be_preferred", str(cfg_path))
            train = [{"id": "train_case"}]
            test = [{"id": "test_case"}]
            return train, test

        orig = ev.tfw.build_datalists
        try:
            ev.tfw.build_datalists = fake_build_datalists  # type: ignore[assignment]
            test_list = ev.load_test_list(args)
        finally:
            ev.tfw.build_datalists = orig  # type: ignore[assignment]

        self.assertEqual(test_list, [{"id": "test_case"}])

    def test_load_test_list_falls_back_to_datalist_when_no_split_cfg(self):
        import json
        import tempfile
        import types
        import scripts.eval_wp5 as ev

        with tempfile.TemporaryDirectory() as td:
            dl_path = os.path.join(td, "test_list.json")
            expected = [{"id": "from_json"}]
            with open(dl_path, "w", encoding="utf-8") as f:
                json.dump(expected, f)

            args = types.SimpleNamespace(
                datalist=dl_path,
                data_root="",
                split_cfg="",
            )
            test_list = ev.load_test_list(args)
        self.assertEqual(test_list, expected)


if __name__ == "__main__":
    unittest.main()
