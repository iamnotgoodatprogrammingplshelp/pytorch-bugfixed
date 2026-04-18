# Owner(s): ["module: dynamo"]

import sys

import torch
import torch._dynamo
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import TestCase, run_tests


class TestGraphBreakSourceInfo(TestCase):
    """Verify `unimplemented()` attaches a `source_info` dict to the
    `Unsupported` exception and appends a `at file:line[:col] in \\`snippet\\``
    pointer to the user-facing message (with an underlined preview on 3.11+)."""

    def test_source_info_attached_by_default(self):
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y + 1

        try:
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor([1.0]))
        except Unsupported as e:
            self.assertIsNotNone(e.source_info, "source_info should be populated")
            self.assertTrue(
                e.source_info["filename"].endswith("test_graph_break_source_info.py"),
                f"wrong filename: {e.source_info['filename']}",
            )
            self.assertIsInstance(e.source_info["lineno"], int)
            if sys.version_info >= (3, 11):
                self.assertIn("col_offset", e.source_info)
        else:
            self.fail("expected an Unsupported graph break")

    def test_source_info_default_does_not_change_message(self):
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y + 1

        try:
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor([1.0]))
        except Unsupported as e:
            self.assertNotIn("test_graph_break_source_info.py:", str(e))
        else:
            self.fail("expected an Unsupported graph break")

    @torch._dynamo.config.patch(graph_break_show_source_info=True)
    def test_source_info_pointer_in_message_when_enabled(self):
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y + 1

        expected_lineno = fn.__code__.co_firstlineno + 2
        try:
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor([2.0]))
        except Unsupported as e:
            self.assertEqual(
                e.source_info["lineno"],
                expected_lineno,
                f"wrong lineno in {e.source_info}",
            )
            self.assertIn(f":{expected_lineno}", str(e))
            self.assertIn("test_graph_break_source_info.py", str(e))
        else:
            self.fail("expected an Unsupported graph break")


if __name__ == "__main__":
    run_tests()
