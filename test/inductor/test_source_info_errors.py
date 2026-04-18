# Owner(s): ["module: inductor"]

import torch
from torch._inductor import lowering
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import run_tests


class TestInductorSourceInfoErrors(TestCase):
    """Verify Inductor surfaces `source_info` in lowering error messages
    (`at <file>:<lineno> in \\`<snippet>\\``) so users can map a compiler
    failure back to the offending line in their own code."""

    def test_lowering_exception_includes_source_info_pointer(self):
        aten_target = torch.ops.aten.sin.default
        original = lowering.lowerings.get(aten_target)
        self.assertIsNotNone(original, "aten.sin lowering should be registered")

        def boom(*args, **kwargs):
            raise RuntimeError("injected lowering failure for source_info test")

        lowering.lowerings[aten_target] = boom
        try:

            def fn(x):
                return x.sin() + 1

            expected_lineno = fn.__code__.co_firstlineno + 1
            try:
                torch.compile(fn, fullgraph=True)(torch.randn(4))
            except Exception as e:
                msg = str(e)
                self.assertIn("injected lowering failure", msg)
                self.assertIn("LoweringException", msg)
                self.assertIn("test_source_info_errors.py", msg)
                self.assertIn(f":{expected_lineno}", msg)
                self.assertIn("x.sin()", msg)
            else:
                self.fail("expected a lowering failure")
        finally:
            lowering.lowerings[aten_target] = original


if __name__ == "__main__":
    run_tests()
