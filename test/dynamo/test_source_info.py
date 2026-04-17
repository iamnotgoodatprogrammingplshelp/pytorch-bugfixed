# Owner(s): ["module: dynamo"]

import sys

import torch
import torch._dynamo
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import EagerAndRecordGraphs


class TestSourceInfo(TestCase):
    """Checks that Dynamo populates node.meta['source_info'] with
    structured source metadata on every captured op node."""

    def _capture(self, fn, *args):
        backend = EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(*args)
        self.assertEqual(len(backend.graphs), 1)
        return backend.graphs[0]

    def test_basic_fields_present(self):
        def fn(x):
            return x.sin().cos() + 1

        gm = self._capture(fn, torch.randn(4))
        op_nodes = [n for n in gm.graph.nodes if n.op != "placeholder" and n.op != "output"]
        self.assertTrue(len(op_nodes) > 0)
        for node in op_nodes:
            info = node.meta.get("source_info")
            self.assertIsNotNone(info, f"source_info missing on {node}")
            for key in ("filename", "function_name", "inline_depth", "node_name", "lineno"):
                self.assertIn(key, info)
            self.assertEqual(info["node_name"], node.name)
            self.assertEqual(info["function_name"], "fn")
            self.assertTrue(info["filename"].endswith("test_source_info.py"))
            self.assertIsInstance(info["inline_depth"], int)

    def test_source_line_matches_user_code(self):
        def fn(x):
            y = x.sin()
            z = y.cos()
            return z + 1

        gm = self._capture(fn, torch.randn(4))
        source_lines = {
            n.meta["source_info"].get("source_line", "")
            for n in gm.graph.nodes
            if n.op == "call_function" and "source_info" in n.meta
        }
        joined = "\n".join(source_lines)
        self.assertIn("x.sin()", joined)
        self.assertIn("y.cos()", joined)

    def test_lineno_is_within_function(self):
        def fn(x):
            return x * 2

        gm = self._capture(fn, torch.randn(4))
        fn_firstlineno = fn.__code__.co_firstlineno
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            info = node.meta["source_info"]
            self.assertGreaterEqual(info["lineno"], fn_firstlineno)
            self.assertLess(info["lineno"], fn_firstlineno + 10)

    def test_inline_depth_tracks_inlining(self):
        def helper(x):
            return x.relu()

        def fn(x):
            return helper(x) + 1

        gm = self._capture(fn, torch.randn(4))
        infos = [
            n.meta["source_info"]
            for n in gm.graph.nodes
            if n.op == "call_function" and "source_info" in n.meta
        ]
        self.assertTrue(infos, "expected at least one call_function node")
        depths = {info["inline_depth"] for info in infos}
        # Outer fn ops at depth 0, inlined helper ops at depth >= 1.
        self.assertIn(0, depths)
        self.assertTrue(
            any(d >= 1 for d in depths),
            f"expected some inlined nodes (depth>=1), got depths={depths}",
        )

    def test_positions_present_on_311plus(self):
        if sys.version_info < (3, 11):
            self.skipTest("bytecode positions require Python 3.11+")

        def fn(x):
            return x.sin()

        gm = self._capture(fn, torch.randn(4))
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            info = node.meta["source_info"]
            self.assertIn("col_offset", info)
            self.assertIn("end_col_offset", info)
            self.assertIn("end_lineno", info)

    def test_source_info_survives_copy_meta_fields(self):
        import torch.fx.proxy

        self.assertIn("source_info", torch.fx.proxy._COPY_META_FIELDS)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
