"""Minimal, provenance-preserving graph for an Euler 2D surface export.

Only the ordered boundary contour is represented.  The Parquet export does
not contain volume-cell adjacency, so this module intentionally creates no
volumetric edges and does not infer them from coordinates or distances.

Changes with respect to the first version (all backward compatible):

1. Inputs, targets and diagnostics are separated per node, so that solver
   outputs (Cp, pressure) cannot leak into the model inputs by accident.
   The full Parquet row is still kept in ``attributes`` for provenance.
2. Reverse contour edges are available, so message passing can run in both
   directions along the contour (the usual convention for undirected graphs
   in message-passing GNNs).  Forward edges keep their original meaning.
3. Every edge carries geometric features: relative displacement between the
   two face centres, its length and the signed turning angle between the two
   outward normals (the scalar and bivector parts of the geometric product of
   the normals).
4. The contour is validated on construction: counter-clockwise orientation,
   outward normals consistent with the contour tangent, closing gap and
   duplicated face centres.  Hard violations raise ``ValueError``; the
   numbers are stored in ``validation`` for inspection.
5. Flow conditions and global coefficients are attached to the graph as
   separate graph-level inputs and targets.
6. ``to_arrays`` exports node/edge features and ``edge_index`` as NumPy
   arrays in the layout used by common GNN libraries (2 x E edge index).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .loader import Euler2DDataset

# ---------------------------------------------------------------------------
# Column roles (contract euler2d-contract-v0.1, surface/case_<id>.parquet)
# ---------------------------------------------------------------------------
#: Geometric, solver-independent quantities: allowed as model inputs.
INPUT_COLUMNS = ("x_over_c", "y_over_c", "s_over_c", "normal_x", "normal_y",
                 "segment_length_m", "side")
#: Solver outputs at the wall face: model targets, never inputs.
TARGET_COLUMNS = ("Cp", "pressure_gauge_Pa", "pressure_abs_Pa")
#: Values of the wall-adjacent cell (NOT face values): diagnostics only.
DIAGNOSTIC_COLUMNS = ("Cp_adjacent_cell", "mach_adjacent_cell")
#: Graph-level quantities taken from cases.csv.
GRAPH_INPUT_KEYS = ("mach_inf", "alpha_deg")
GRAPH_TARGET_KEYS = ("CL", "CD")

#: Names of the numeric node features produced by ``to_arrays``.
NODE_FEATURE_NAMES = ("x_over_c", "y_over_c", "s_over_c", "normal_x", "normal_y",
                      "segment_length_over_c", "side_upper")
NODE_TARGET_NAMES = ("Cp",)
EDGE_FEATURE_NAMES = ("dx_over_c", "dy_over_c", "length_over_c",
                      "normal_dot", "normal_wedge", "normal_turn_rad")

# Validation thresholds
_MIN_NORMAL_TANGENT_ALIGNMENT = 0.9   # cos(~25 deg) between normal and rotated tangent
_MAX_CLOSING_GAP_RATIO = 3.0          # closing edge length / median edge length
_DUPLICATE_TOL_M = 1e-12


@dataclass(frozen=True)
class SurfaceNode:
    """One exported surface face, retaining identifiers and physical fields."""

    node_id: str
    case_id: str
    geometry_id: str
    entity_id: str
    contour_index: int
    x_m: float
    y_m: float
    normal_x: float
    normal_y: float
    segment_length_m: float
    attributes: Mapping[str, Any]
    inputs: Mapping[str, Any] = field(default_factory=dict)
    targets: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SurfaceEdge:
    """A directed edge justified by consecutive documented contour order."""

    source: str
    target: str
    contour_index: int
    connectivity: str = "boundary_contour"
    direction: str = "forward"
    dx_m: float = 0.0
    dy_m: float = 0.0
    length_m: float = 0.0
    normal_dot: float = 1.0      # n_s . n_t   (scalar part of n_s n_t)
    normal_wedge: float = 0.0    # n_s ^ n_t   (bivector part, e1e2 coefficient)

    @property
    def normal_turn_rad(self) -> float:
        """Signed angle from the source normal to the target normal."""
        return math.atan2(self.normal_wedge, self.normal_dot)


@dataclass(frozen=True)
class SurfaceGraph:
    """Graph with boundary connectivity only and explicit Parquet provenance."""

    case_id: str
    geometry_id: str
    source_parquet: str
    nodes: tuple[SurfaceNode, ...]
    contour_edges: tuple[SurfaceEdge, ...]
    volumetric_edges: tuple[SurfaceEdge, ...] = ()
    reverse_edges: tuple[SurfaceEdge, ...] = ()
    chord_ref_m: float = 1.0
    graph_inputs: Mapping[str, Any] = field(default_factory=dict)
    graph_targets: Mapping[str, Any] = field(default_factory=dict)
    status: str = ""
    validation: Mapping[str, Any] = field(default_factory=dict)

    @property
    def connectivity_kinds(self) -> tuple[str, ...]:
        kinds = []
        if self.contour_edges:
            kinds.append("boundary_contour")
        if self.reverse_edges:
            kinds.append("boundary_contour_reverse")
        return tuple(kinds)

    def edges(self, bidirectional: bool = True) -> tuple[SurfaceEdge, ...]:
        """Forward contour edges, plus their reverses when ``bidirectional``."""
        return self.contour_edges + (self.reverse_edges if bidirectional else ())

    def summary(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "geometry_id": self.geometry_id,
            "status": self.status,
            "source_parquet": self.source_parquet,
            "node_count": len(self.nodes),
            "boundary_edge_count": len(self.contour_edges),
            "reverse_edge_count": len(self.reverse_edges),
            "volumetric_edge_count": len(self.volumetric_edges),
            "connectivity": list(self.connectivity_kinds),
            "graph_inputs": dict(self.graph_inputs),
            "graph_targets": dict(self.graph_targets),
            "validation": dict(self.validation),
            "first_nodes": [
                {
                    "node_id": node.node_id,
                    "entity_id": node.entity_id,
                    "contour_index": node.contour_index,
                    "x_m": node.x_m,
                    "y_m": node.y_m,
                    "Cp": node.targets.get("Cp"),
                }
                for node in self.nodes[:3]
            ],
        }

    def to_arrays(self, bidirectional: bool = True) -> dict[str, Any]:
        """Numeric arrays for a GNN: features, targets and a 2 x E edge index.

        Node inputs never include solver outputs; targets are returned apart.
        """
        import numpy as np

        position = {node.node_id: i for i, node in enumerate(self.nodes)}
        c = self.chord_ref_m
        x = np.array([[float(n.inputs["x_over_c"]), float(n.inputs["y_over_c"]),
                       float(n.inputs["s_over_c"]), n.normal_x, n.normal_y,
                       n.segment_length_m / c, 1.0 if n.inputs["side"] == "upper" else 0.0]
                      for n in self.nodes])
        y = np.array([[float(n.targets["Cp"])] for n in self.nodes])
        edges = self.edges(bidirectional)
        edge_index = np.array([[position[e.source] for e in edges],
                               [position[e.target] for e in edges]], dtype=np.int64)
        edge_attr = np.array([[e.dx_m / c, e.dy_m / c, e.length_m / c,
                               e.normal_dot, e.normal_wedge, e.normal_turn_rad] for e in edges])
        return {
            "node_features": x, "node_feature_names": NODE_FEATURE_NAMES,
            "node_targets": y, "node_target_names": NODE_TARGET_NAMES,
            "edge_index": edge_index, "edge_features": edge_attr,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "graph_inputs": dict(self.graph_inputs), "graph_targets": dict(self.graph_targets),
        }

    def continuity_report(self):
        """Run the non-topological center-gap continuity diagnostic."""
        from .geometry import check_contour_continuity

        return check_contour_continuity(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _pick(row: Mapping[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    return {name: row[name] for name in columns if name in row}


def _make_edge(a: SurfaceNode, b: SurfaceNode, direction: str, connectivity: str) -> SurfaceEdge:
    return SurfaceEdge(
        source=a.node_id, target=b.node_id, contour_index=a.contour_index,
        connectivity=connectivity, direction=direction,
        dx_m=b.x_m - a.x_m, dy_m=b.y_m - a.y_m, length_m=math.hypot(b.x_m - a.x_m, b.y_m - a.y_m),
        normal_dot=a.normal_x * b.normal_x + a.normal_y * b.normal_y,
        normal_wedge=a.normal_x * b.normal_y - a.normal_y * b.normal_x,
    )


def validate_contour(nodes: tuple[SurfaceNode, ...], case_id: str) -> dict[str, Any]:
    """Check orientation, normals and closure of an ordered closed contour.

    - Orientation: signed polygon area (shoelace formula) must be positive,
      i.e. the contour is traversed counter-clockwise.
    - Normals: for a counter-clockwise contour, the outward normal equals the
      unit tangent rotated by -90 deg, (t_y, -t_x).  The tangent is taken by
      central differences of the face centres.
    - Closure: the closing edge (last -> first face) must not be much longer
      than a typical edge, otherwise the contour is open or misordered.
    """
    n = len(nodes)
    xs = [node.x_m for node in nodes]
    ys = [node.y_m for node in nodes]
    area = 0.5 * sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))

    lengths = sorted(math.hypot(xs[(i + 1) % n] - xs[i], ys[(i + 1) % n] - ys[i]) for i in range(n))
    median_len = lengths[n // 2]
    closing = math.hypot(xs[0] - xs[-1], ys[0] - ys[-1])
    duplicates = sum(1 for length in lengths if length < _DUPLICATE_TOL_M)

    min_align, misaligned = 1.0, 0
    for i in range(n):
        tx = xs[(i + 1) % n] - xs[i - 1]
        ty = ys[(i + 1) % n] - ys[i - 1]
        norm = math.hypot(tx, ty)
        if norm == 0.0:
            continue
        align = (ty / norm) * nodes[i].normal_x + (-tx / norm) * nodes[i].normal_y
        min_align = min(min_align, align)
        if align < _MIN_NORMAL_TANGENT_ALIGNMENT:
            misaligned += 1

    report = {
        "signed_area_m2": area, "orientation": "ccw" if area > 0 else "cw",
        "closing_gap_ratio": closing / median_len if median_len > 0 else float("inf"),
        "min_normal_tangent_alignment": min_align, "misaligned_normals": misaligned,
        "duplicate_centres": duplicates,
    }
    if area <= 0:
        raise ValueError(f"{case_id}: contour is clockwise (signed area {area:.3e} m2); expected ccw")
    if misaligned > 0:
        raise ValueError(f"{case_id}: {misaligned} normals inconsistent with the contour tangent")
    if report["closing_gap_ratio"] > _MAX_CLOSING_GAP_RATIO:
        raise ValueError(f"{case_id}: closing gap is {report['closing_gap_ratio']:.2f} x the median edge")
    if duplicates:
        raise ValueError(f"{case_id}: {duplicates} duplicated consecutive face centres")
    return report


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_surface_graph(dataset: Euler2DDataset, case_id: str) -> SurfaceGraph:
    """Build only the contour graph explicitly supported by the source data."""
    case = dataset.case(case_id)
    if case["status"] == "failed":
        raise ValueError(f"cannot build a surface graph for failed case {case_id}")
    rows = dataset.surface_rows(case_id)
    if not rows:
        raise ValueError(f"cannot build a surface graph with no rows: {case_id}")
    indices = [int(row["contour_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"{case_id}: contour_index must be contiguous and ordered")
    missing = [c for c in INPUT_COLUMNS + ("Cp",) if c not in rows[0]]
    if missing:
        raise ValueError(f"{case_id}: missing surface columns {missing}")

    nodes: list[SurfaceNode] = []
    for row in rows:
        contour_index = int(row["contour_index"])
        nodes.append(
            SurfaceNode(
                node_id=f"{case_id}:face:{contour_index}",
                case_id=case_id,
                geometry_id=case["geometry_id"],
                entity_id=str(row["entity_id"]),
                contour_index=contour_index,
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                normal_x=float(row["normal_x"]),
                normal_y=float(row["normal_y"]),
                segment_length_m=float(row["segment_length_m"]),
                attributes=dict(row),
                inputs=_pick(row, INPUT_COLUMNS),
                targets=_pick(row, TARGET_COLUMNS),
                diagnostics=_pick(row, DIAGNOSTIC_COLUMNS),
            )
        )
    node_tuple = tuple(nodes)
    validation = validate_contour(node_tuple, case_id)

    count = len(node_tuple)
    forward = tuple(_make_edge(node_tuple[i], node_tuple[(i + 1) % count], "forward", "boundary_contour")
                    for i in range(count))
    reverse = tuple(_make_edge(node_tuple[(i + 1) % count], node_tuple[i], "reverse", "boundary_contour_reverse")
                    for i in range(count))

    source = Path(dataset.surface_path(case_id))
    try:
        source_name = source.relative_to(dataset.root).as_posix()
    except ValueError:
        source_name = source.as_posix()
    return SurfaceGraph(
        case_id=case_id,
        geometry_id=case["geometry_id"],
        source_parquet=source_name,
        nodes=node_tuple,
        contour_edges=forward,
        reverse_edges=reverse,
        chord_ref_m=_as_float(case.get("chord_ref_m", 1.0)),
        graph_inputs={k: _as_float(case[k]) for k in GRAPH_INPUT_KEYS if k in case},
        graph_targets={k: _as_float(case[k]) for k in GRAPH_TARGET_KEYS if k in case},
        status=str(case["status"]),
        validation=validation,
    )
