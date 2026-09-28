"""Explicit indicator dependency DAG. Cycles are errors, not implicit loops."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from solardss.engine.contracts import Dependency, DependencyKind, IndicatorNode
from solardss.engine.errors import CircularDependencyError, IndicatorEngineError
from solardss.exceptions import UnknownIndicatorError


class IndicatorGraph:
    """Frozen DAG of indicator nodes. Missing nodes are not invented."""

    def __init__(self, nodes: Sequence[IndicatorNode]) -> None:
        by_id: dict[str, IndicatorNode] = {}
        for node in nodes:
            if node.indicator_id in by_id:
                raise IndicatorEngineError(f"duplicate indicator node {node.indicator_id!r}")
            by_id[node.indicator_id] = node
        self._nodes = by_id
        self._cycle_check()

    def node(self, indicator_id: str) -> IndicatorNode:
        """Return the node for ``indicator_id``."""

        try:
            return self._nodes[indicator_id]
        except KeyError as exc:
            raise UnknownIndicatorError(
                f"indicator {indicator_id!r} is not in the dependency graph"
            ) from exc

    def has(self, indicator_id: str) -> bool:
        return indicator_id in self._nodes

    def nodes(self) -> tuple[IndicatorNode, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    def dependencies_of(self, indicator_id: str) -> tuple[Dependency, ...]:
        return self.node(indicator_id).dependencies

    def indicator_dependencies(
        self, indicator_id: str, *, required_only: bool = False
    ) -> tuple[str, ...]:
        """Indicator ids this node lists. Source-type edges are omitted."""

        found: list[str] = []
        for dep in self.node(indicator_id).dependencies:
            if dep.kind is not DependencyKind.INDICATOR:
                continue
            if required_only and not dep.required:
                continue
            found.append(dep.target)
        return tuple(found)

    def expand(
        self,
        requested: Sequence[str],
        *,
        extra_indicator_deps: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[str, ...]:
        """Closure of requested ids plus explicit indicator edges.

        Source-type dependencies do not add extra fetches. Optional indicator
        edges are included only when listed in ``extra_indicator_deps``
        (typically calculator inputs).
        """

        extras = extra_indicator_deps or {}
        seen: set[str] = set()
        stack = list(requested)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            self.node(current)
            seen.add(current)
            for dep_id in self.indicator_dependencies(current, required_only=True):
                stack.append(dep_id)
            for dep_id in extras.get(current, ()):
                stack.append(dep_id)
        return tuple(sorted(seen))

    def topological_order(self, indicator_ids: Sequence[str]) -> tuple[str, ...]:
        """Kahn order of ``indicator_ids``. Raises on a cycle in the subgraph."""

        wanted = set(indicator_ids)
        for item in wanted:
            self.node(item)
        incoming: dict[str, int] = {item: 0 for item in wanted}
        edges: dict[str, list[str]] = {item: [] for item in wanted}
        for item in wanted:
            for dep_id in self.indicator_dependencies(item):
                if dep_id not in wanted:
                    continue
                edges[dep_id].append(item)
                incoming[item] += 1
        ready = sorted(item for item, count in incoming.items() if count == 0)
        ordered: list[str] = []
        while ready:
            node_id = ready.pop(0)
            ordered.append(node_id)
            for child in sorted(edges[node_id]):
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(ordered) != len(wanted):
            leftover = tuple(sorted(item for item in wanted if item not in set(ordered)))
            cycle = self._cycle_among(leftover)
            raise CircularDependencyError(
                f"circular indicator dependencies: {' -> '.join(cycle)}",
                cycle=cycle,
            )
        return tuple(ordered)

    def overlay(self, nodes: Sequence[IndicatorNode]) -> IndicatorGraph:
        """Return a new graph with ``nodes`` replacing matching ids."""

        merged = {item.indicator_id: item for item in self.nodes()}
        for node in nodes:
            merged[node.indicator_id] = node
        return IndicatorGraph(tuple(merged[key] for key in sorted(merged)))

    def _cycle_check(self) -> None:
        self.topological_order(tuple(self._nodes))

    def _cycle_among(self, leftover: Sequence[str]) -> tuple[str, ...]:
        remaining = set(leftover)
        path: list[str] = []
        visiting: set[str] = set()
        found: list[str] = []

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                start = path.index(node_id)
                found.extend([*path[start:], node_id])
                return True
            if node_id not in remaining:
                return False
            visiting.add(node_id)
            path.append(node_id)
            for dep_id in self.indicator_dependencies(node_id):
                if dep_id in remaining and visit(dep_id):
                    return True
            path.pop()
            visiting.remove(node_id)
            remaining.discard(node_id)
            return False

        for item in leftover:
            if visit(item):
                return tuple(found)
        return tuple(leftover)


def merge_nodes(
    base: Iterable[IndicatorNode], overrides: Iterable[IndicatorNode]
) -> tuple[IndicatorNode, ...]:
    """Replace base nodes that share an id with ``overrides``."""

    merged = {item.indicator_id: item for item in base}
    for node in overrides:
        merged[node.indicator_id] = node
    return tuple(merged[key] for key in sorted(merged))
