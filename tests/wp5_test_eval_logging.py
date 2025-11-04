import os
import sys
import tempfile
import unittest


class TestEvalLogging(unittest.TestCase):
    def test_eval_logging_writes_file(self):
        import scripts.eval_wp5 as ev
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "eval_run")
            ev._init_eval_logging(out_dir=__import__('pathlib').Path(out_dir), enable=True, filename="eval.log")
            print("hello-eval-log")
            logp = os.path.join(out_dir, "eval.log")
            with open(logp, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("hello-eval-log", content)


if __name__ == "__main__":
    unittest.main()

