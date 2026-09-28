"""Indicator registry: lookup, filtering, and mode-aware resolution.

The registry does not fetch datasets. It only describes the vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from solardss.exceptions import UnknownIndicatorError
from solardss.indicators.contracts import (
    DecisionMode,
    IndicatorCategory,
    IndicatorSpec,
    ModeContext,
    ProjectType,
    RequirementLevel,
    ResolvedIndicator,
    SiteSelectionScope,
)
from solardss.indicators.errors import IndicatorRegistryError
from solardss.indicators.ids import IndicatorId


def _as_mode(value: DecisionMode | str) -> DecisionMode:
    if isinstance(value, DecisionMode):
        return value
    try:
        return DecisionMode(value)
    except ValueError as exc:
        raise IndicatorRegistryError(f"Unknown decision mode {value!r}") from exc


def _as_project_type(value: ProjectType | str) -> ProjectType:
    if isinstance(value, ProjectType):
        return value
    try:
        return ProjectType(value)
    except ValueError as exc:
        raise IndicatorRegistryError(f"Unknown project type {value!r}") from exc


def _as_category(value: IndicatorCategory | str) -> IndicatorCategory:
    if isinstance(value, IndicatorCategory):
        return value
    try:
        return IndicatorCategory(value)
    except ValueError as exc:
        raise IndicatorRegistryError(f"Unknown indicator category {value!r}") from exc


def _as_scope(value: SiteSelectionScope | str) -> SiteSelectionScope:
    if isinstance(value, SiteSelectionScope):
        return value
    try:
        return SiteSelectionScope(value)
    except ValueError as exc:
        raise IndicatorRegistryError(f"Unknown site-selection scope {value!r}") from exc


class IndicatorRegistry:
    """Immutable lookup table of indicator specifications."""

    def __init__(self, definitions: Sequence[IndicatorSpec]) -> None:
        by_id: dict[str, IndicatorSpec] = {}
        aliases: dict[str, str] = {}
        seen_names: dict[str, str] = {}
        for spec in definitions:
            if spec.id in by_id:
                raise IndicatorRegistryError(f"Duplicate indicator id {spec.id!r}")
            by_id[spec.id] = spec
            if spec.name in seen_names:
                raise IndicatorRegistryError(
                    f"Duplicate indicator name {spec.name!r} for {spec.id!r} "
                    f"and {seen_names[spec.name]!r}"
                )
            seen_names[spec.name] = spec.id
            for alias in spec.aliases:
                if alias == spec.id or alias in by_id or alias in aliases:
                    raise IndicatorRegistryError(f"Alias {alias!r} collides with an id or alias")
                aliases[alias] = spec.id
        expected = {member.value for member in IndicatorId}
        actual = set(by_id)
        if expected != actual:
            missing = tuple(sorted(expected - actual))
            extra = tuple(sorted(actual - expected))
            raise IndicatorRegistryError(
                f"Registry ids do not match IndicatorId enum. missing={missing} extra={extra}"
            )
        self._by_id = by_id
        self._aliases = aliases
        self._all = tuple(sorted(by_id.values(), key=lambda spec: spec.id))

    def _canonical_id(self, indicator_id: str | IndicatorId) -> str:
        key = indicator_id.value if isinstance(indicator_id, IndicatorId) else indicator_id
        if key in self._by_id:
            return key
        if key in self._aliases:
            return self._aliases[key]
        raise UnknownIndicatorError(f"Unknown indicator {key!r}")

    def get(self, indicator_id: str | IndicatorId) -> IndicatorSpec:
        """Return the single registered definition for ``indicator_id`` or an alias."""

        return self._by_id[self._canonical_id(indicator_id)]

    def list_all(self) -> tuple[IndicatorSpec, ...]:
        """All definitions, sorted by id."""

        return self._all

    def by_category(self, category: IndicatorCategory | str) -> tuple[IndicatorSpec, ...]:
        """Definitions in one vocabulary section."""

        wanted = _as_category(category)
        return tuple(spec for spec in self._all if spec.category is wanted)

    def for_mode(
        self,
        mode: DecisionMode | str,
        *,
        scope: SiteSelectionScope | str | None = None,
    ) -> tuple[IndicatorSpec, ...]:
        """Definitions applicable to a decision mode, optionally a site-selection scope."""

        wanted = _as_mode(mode)
        wanted_scope = _as_scope(scope) if scope is not None else None
        selected: list[IndicatorSpec] = []
        for spec in self._all:
            if wanted not in spec.applicable_decision_modes:
                continue
            if (
                wanted is DecisionMode.SITE_SELECTION
                and wanted_scope is not None
                and wanted_scope not in spec.applicable_site_selection_scopes
            ):
                continue
            selected.append(spec)
        return tuple(selected)

    def for_project_type(self, project_type: ProjectType | str) -> tuple[IndicatorSpec, ...]:
        """Definitions applicable to a project class."""

        wanted = _as_project_type(project_type)
        return tuple(spec for spec in self._all if wanted in spec.applicable_project_types)

    def required_for(
        self,
        mode: DecisionMode | str,
        project_type: ProjectType | str | None = None,
        *,
        scope: SiteSelectionScope | str | None = None,
    ) -> tuple[IndicatorSpec, ...]:
        """Definitions that are required in the given mode (and project type, if set)."""

        wanted_mode = _as_mode(mode)
        wanted_project = _as_project_type(project_type) if project_type is not None else None
        required: list[IndicatorSpec] = []
        for spec in self.for_mode(wanted_mode, scope=scope):
            if wanted_project is not None and wanted_project not in spec.applicable_project_types:
                continue
            resolved = self.resolve(spec.id, mode=wanted_mode, project_type=wanted_project)
            if resolved.required_or_optional is RequirementLevel.REQUIRED:
                required.append(spec)
        return tuple(required)

    def resolve(
        self,
        indicator_id: str | IndicatorId,
        *,
        mode: DecisionMode | str,
        project_type: ProjectType | str | None = None,
    ) -> ResolvedIndicator:
        """Apply mode context to a single registered definition.

        ``PV_LOCATION`` is one definition. Site selection resolves it to a
        controllable decision variable; fixed-site resolves it to fixed context.
        """

        spec = self.get(indicator_id)
        wanted_mode = _as_mode(mode)
        wanted_project = _as_project_type(project_type) if project_type is not None else None
        if wanted_mode not in spec.applicable_decision_modes:
            raise IndicatorRegistryError(
                f"{spec.id} is not applicable to decision mode {wanted_mode.value}"
            )
        if wanted_project is not None and wanted_project not in spec.applicable_project_types:
            raise IndicatorRegistryError(
                f"{spec.id} is not applicable to project type {wanted_project.value}"
            )
        context = _context_for_mode(spec.mode_contexts, wanted_mode)
        role = context.role if context is not None else spec.role
        controllability = context.controllability if context is not None else spec.controllability
        required = spec.required_or_optional
        if context is not None and context.required_or_optional is not None:
            required = context.required_or_optional
        return ResolvedIndicator(
            definition=spec,
            mode=wanted_mode,
            project_type=wanted_project,
            role=role,
            controllability=controllability,
            required_or_optional=required,
        )


def _context_for_mode(
    contexts: Sequence[ModeContext],
    mode: DecisionMode,
) -> ModeContext | None:
    for context in contexts:
        if context.mode is mode:
            return context
    return None


def build_registry(definitions: Iterable[IndicatorSpec]) -> IndicatorRegistry:
    """Validate and freeze a collection of definitions."""

    return IndicatorRegistry(tuple(definitions))


def mapping_from_registry(registry: IndicatorRegistry) -> Mapping[str, IndicatorSpec]:
    """Canonical id → spec mapping (no aliases)."""

    return {spec.id: spec for spec in registry.list_all()}
