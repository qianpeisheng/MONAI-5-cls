import subprocess
import sys
import unittest


class TestDeprecatedEvaluators(unittest.TestCase):
    def test_old_semantics_script_exits(self):
        proc = subprocess.run([sys.executable, "scripts/eval_wp5_old_semantics.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("DEPRECATED", proc.stderr + proc.stdout)

    def test_present_only_script_exits(self):
        proc = subprocess.run([sys.executable, "scripts/verify_present_only_eval.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("DEPRECATED", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()

