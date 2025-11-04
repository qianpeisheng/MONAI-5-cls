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


@unittest.skipUnless(_torch_available, "torch not available")
class TestEvalScriptCLI(unittest.TestCase):
    def test_eval_cli_no_empty_pair_policy_and_has_logging(self):
        import importlib
        import scripts.eval_wp5 as ev
        # The script should expose get_parser for testing
        parser = ev.get_parser()
        opts = {a.dest for a in parser._actions}
        self.assertNotIn("empty_pair_policy", opts)
        self.assertIn("log_to_file", opts)
        self.assertIn("log_file_name", opts)


if __name__ == "__main__":
    unittest.main()
