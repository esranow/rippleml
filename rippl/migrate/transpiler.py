"""
Auto-Migrate: AST-based transpiler for DeepXDE and Modulus scripts.
Parses foreign framework scripts and maps to rippl native API.
Does NOT execute foreign code — static analysis only.
"""
import ast
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExtractedPDE:
    pde_fn_source: str = ""
    domain_type: str = ""        # "interval", "rectangle", "sphere"
    spatial_bounds: list = field(default_factory=list)
    time_range: Optional[tuple] = None
    bcs: list = field(default_factory=list)
    ics: list = field(default_factory=list)
    num_domain: int = 5000
    num_boundary: int = 1000
    num_initial: int = 1000
    framework: str = ""          # "deepxde", "modulus", "sciann"

class DeepXDETranspiler(ast.NodeVisitor):
    """
    Parses DeepXDE scripts via Python AST.
    Extracts: PDE function, geometry, BCs, ICs, collocation counts.
    Maps to rippl System + rp.run() call.
    """
    def __init__(self):
        self.extracted = ExtractedPDE(framework="deepxde")
        self._source_lines = []

    def parse(self, source: str) -> ExtractedPDE:
        self._source_lines = source.splitlines()
        tree = ast.parse(source)
        self.visit(tree)
        return self.extracted

    def visit_FunctionDef(self, node):
        # Detect PDE function: signature (x, y) or (x, u) pattern
        if len(node.args.args) == 2:
            self.extracted.pde_fn_source = ast.get_source_segment(
                "\n".join(self._source_lines), node
            ) or ""
        self.generic_visit(node)

    def visit_Call(self, node):
        fn = self._get_call_name(node)
        # GeometryXTime
        if "GeometryXTime" in fn:
            self.extracted.domain_type = "spacetime"
        # Interval
        if "Interval" in fn and len(node.args) >= 2:
            lo = self._eval_const(node.args[0])
            hi = self._eval_const(node.args[1])
            if lo is not None and hi is not None:
                self.extracted.spatial_bounds.append((lo, hi))
        # TimeDomain
        if "TimeDomain" in fn and len(node.args) >= 2:
            t0 = self._eval_const(node.args[0])
            t1 = self._eval_const(node.args[1])
            if t0 is not None and t1 is not None:
                self.extracted.time_range = (t0, t1)
        # DirichletBC
        if "DirichletBC" in fn:
            self.extracted.bcs.append({"type": "dirichlet"})
        # IC
        if fn.endswith(".IC") or "icbc.IC" in fn:
            self.extracted.ics.append({"type": "initial"})
        # num_domain
        for kw in node.keywords:
            if kw.arg == "num_domain":
                v = self._eval_const(kw.value)
                if v: self.extracted.num_domain = int(v)
            if kw.arg == "num_boundary":
                v = self._eval_const(kw.value)
                if v: self.extracted.num_boundary = int(v)
            if kw.arg == "num_initial":
                v = self._eval_const(kw.value)
                if v: self.extracted.num_initial = int(v)
        self.generic_visit(node)

    def _get_call_name(self, node) -> str:
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
        return ""

    def _eval_const(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self._eval_const(node.operand)
            return -v if v is not None else None
        return None

    def to_rippl_script(self) -> str:
        """Generate rippl equivalent script from extracted PDE."""
        e = self.extracted
        bounds = list(e.spatial_bounds)
        if e.time_range:
            bounds.append(list(e.time_range))
        lines = [
            "import rippl as rp",
            "import rippl.nn as rnn",
            "import torch",
            "",
            "# --- Auto-migrated from DeepXDE ---",
            "# Review PDE function and adjust operator composition",
            "",
            f"domain = rp.Domain(",
            f"    spatial_dims={len(e.spatial_bounds)},",
            f"    bounds={bounds}",
            f")",
            "",
            "# TODO: Replace with rippl operator composition",
            "# equation = rp.Equation([rp.TimeDerivative(1), rp.Diffusion(alpha=...)])",
            "",
            f"model = rnn.MLP(in_dim={len(bounds)}, out_dim=1, hidden=50, layers=4)",
            "",
            "result = rp.run(",
            "    domain=domain,",
            "    equation=equation,",
            "    model=model,",
            f"    batch_size={e.num_domain},",
            ")",
        ]
        return "\n".join(lines)


def migrate(source: str, framework: str = "auto") -> str:
    """
    Main entry point. Detect framework and transpile.
    
    Args:
        source: source code string of foreign script
        framework: "deepxde", "modulus", "sciann", or "auto" (detect)
    
    Returns:
        rippl equivalent script as string with TODO markers where
        manual review is required
    
    Raises:
        MigrationError if source cannot be parsed
    """
    if framework == "auto":
        framework = _detect_framework(source)
    if framework == "deepxde":
        t = DeepXDETranspiler()
        extracted = t.parse(source)
        return t.to_rippl_script()
    raise MigrationError(f"Unsupported framework: {framework}. Supported: deepxde")

def _detect_framework(source: str) -> str:
    if "import deepxde" in source or "deepxde" in source:
        return "deepxde"
    if "modulus" in source.lower():
        return "modulus"
    if "sciann" in source.lower():
        return "sciann"
    raise MigrationError("Cannot detect source framework. Pass framework= explicitly.")

class MigrationError(ValueError): pass
