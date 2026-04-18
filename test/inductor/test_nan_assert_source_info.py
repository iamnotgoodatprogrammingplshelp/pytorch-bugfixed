# Owner(s): ["module: inductor"]

"""Slice 4 — verify `nan_asserts_show_source_info` threads source_info from
FX node metadata into the NaN/Inf assertion message the Triton codegen emits."""

from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import run_tests


class _FakeIRNode:
    def __init__(self, infos):
        self._infos = infos

    def get_source_infos(self):
        return self._infos


class _FakeSchedulerNode:
    def __init__(self, ir_node):
        self.node = ir_node


class _FakeFeatures:
    def __init__(self, nodes):
        self._nodes = nodes

    def scheduler_nodes(self):
        return self._nodes


class _FakeKernel:
    """Minimal stand-in for the SIMD kernel — only exposes what the helper
    reads (`self.features.scheduler_nodes()`)."""

    def __init__(self, features):
        self.features = features

    _source_info_hint_for_nan_check = (
        # rebind the real method onto the fake class
        __import__(
            "torch._inductor.codegen.triton", fromlist=["TritonKernel"]
        ).TritonKernel._source_info_hint_for_nan_check
    )


class TestNanAssertSourceInfo(TestCase):
    def test_hint_uses_first_available_source_info(self):
        info = {
            "filename": "/tmp/user_code.py",
            "lineno": 42,
            "source_line": "    y = x.sin()",
        }
        kernel = _FakeKernel(_FakeFeatures([_FakeSchedulerNode(_FakeIRNode([info]))]))
        hint = kernel._source_info_hint_for_nan_check()
        self.assertIn("/tmp/user_code.py:42", hint)
        self.assertIn("x.sin()", hint)
        self.assertTrue(hint.startswith("(from user code at "))

    def test_hint_empty_when_no_source_info(self):
        kernel = _FakeKernel(_FakeFeatures([_FakeSchedulerNode(_FakeIRNode([]))]))
        self.assertEqual(kernel._source_info_hint_for_nan_check(), "")

    def test_hint_empty_when_no_features(self):
        kernel = _FakeKernel(None)
        self.assertEqual(kernel._source_info_hint_for_nan_check(), "")

    def test_config_flag_defaults_off(self):
        self.assertFalse(config.nan_asserts_show_source_info)

    def test_hint_skips_info_without_lineno(self):
        bad = {"filename": "/tmp/x.py"}
        good = {"filename": "/tmp/y.py", "lineno": 7, "source_line": "z = 1"}
        kernel = _FakeKernel(
            _FakeFeatures([_FakeSchedulerNode(_FakeIRNode([bad, good]))])
        )
        hint = kernel._source_info_hint_for_nan_check()
        self.assertIn("/tmp/y.py:7", hint)
        self.assertNotIn("/tmp/x.py", hint)


if __name__ == "__main__":
    run_tests()
