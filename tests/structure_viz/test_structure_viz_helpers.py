import re
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Lattice, PeriodicSite, Structure

from pymatviz.enums import ElemColorScheme, SiteCoords
from pymatviz.structure_viz.helpers import (
    NO_SYM_MSG,
    UNIT_CELL_EDGES,
    _angles_to_rotation_matrix,
    draw_bonds,
    draw_site,
    draw_unit_cell,
    draw_vector,
    get_atomic_radii,
    get_elem_colors,
    get_first_matching_site_prop,
    get_image_sites,
    get_site_hover_text,
    get_structures,
    get_subplot_title,
)
from pymatviz.typing import RgbColorType


@pytest.fixture
def mock_figure() -> Any:
    class MockFigure:
        def __init__(self) -> None:
            self.last_trace_kwargs: dict[str, Any] = {}

        def add_scatter(self, *_args: Any, **kwargs: Any) -> None:
            self.last_trace_kwargs = kwargs

        def add_scatter3d(self, *_args: Any, **kwargs: Any) -> None:
            self.last_trace_kwargs = kwargs

    return MockFigure()


@pytest.mark.parametrize(
    ("angles", "expected_shape"),
    [("10x,8y,3z", (3, 3)), ("30x,45y,60z", (3, 3))],
)
def test_angles_to_rotation_matrix(
    angles: str, expected_shape: tuple[int, int]
) -> None:
    rot_matrix = _angles_to_rotation_matrix(angles)
    assert rot_matrix.shape == expected_shape
    assert_allclose(np.linalg.det(rot_matrix), 1.0)


def test_angles_to_rotation_matrix_invalid_input() -> None:
    with pytest.raises(ValueError, match="could not convert string to float: "):
        _angles_to_rotation_matrix("invalid_input")


def test_get_structures(structures: list[Structure]) -> None:
    # Test with single structure
    result = get_structures(structures[0])
    assert isinstance(result, dict)
    assert len(result) == 1
    assert isinstance(next(iter(result.values())), Structure)

    # Test with list of structures
    result = get_structures(structures)
    assert isinstance(result, dict)
    assert len(result) == len(structures)
    assert all(isinstance(s, Structure) for s in result.values())

    # Test with dict of structures
    structs_dict = {f"struct{i}": struct for i, struct in enumerate(structures)}
    result = get_structures(structs_dict)
    assert isinstance(result, dict)
    assert len(result) == len(structures)
    assert all(isinstance(s, Structure) for s in result.values())


@pytest.mark.parametrize(
    ("elem_colors", "expected_result"),
    [
        (ElemColorScheme.jmol, dict),
        (ElemColorScheme.vesta, dict),
        ({"Si": "#FF0000", "O": "#00FF00"}, dict),
    ],
)
def test_get_elem_colors(
    elem_colors: ElemColorScheme | dict[str, RgbColorType], expected_result: type
) -> None:
    colors = get_elem_colors(elem_colors)
    assert isinstance(colors, expected_result)
    if isinstance(elem_colors, dict):
        assert colors == elem_colors
    else:
        assert "Si" in colors
        assert "O" in colors


def test_get_elem_colors_invalid_input() -> None:
    err_msg = f"colors must be a dict or one of ('{', '.join(ElemColorScheme)}')"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        get_elem_colors("invalid_input")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("atomic_radii", "expected_type"),
    [(None, dict), (1.5, dict), ({"Si": 0.3, "O": 0.2}, dict)],
)
def test_get_atomic_radii(
    atomic_radii: float | dict[str, float] | None, expected_type: type
) -> None:
    radii = get_atomic_radii(atomic_radii)
    assert isinstance(radii, expected_type)

    if atomic_radii is None or isinstance(atomic_radii, float):
        assert "Si" in radii
        assert "O" in radii
        if isinstance(atomic_radii, float):
            assert radii["Si"] == pytest.approx(1.11 * atomic_radii)
    elif isinstance(atomic_radii, dict):
        assert radii == atomic_radii


def test_get_image_sites(structures: list[Structure]) -> None:
    structure = structures[0]
    site = structure[0]
    lattice = structure.lattice

    # Test with default tolerance
    image_atoms = get_image_sites(site, lattice)
    assert isinstance(image_atoms, np.ndarray)
    assert image_atoms.ndim in (1, 2)  # Allow both 1D and 2D arrays
    if image_atoms.size > 0:
        assert image_atoms.shape[1] == 3  # Each image atom should have 3 coordinates

    # Test with custom tolerance
    image_atoms = get_image_sites(site, lattice, tol=0.1)
    assert isinstance(image_atoms, np.ndarray)
    assert image_atoms.ndim in (1, 2)
    if image_atoms.size > 0:
        assert image_atoms.shape[1] == 3


@pytest.mark.parametrize(
    ("lattice", "site_coords", "expected_images"),
    [
        (Lattice.cubic(1), [0, 0, 0], 7),
        (Lattice.cubic(1), [0, 0.5, 0], 3),
        (Lattice.hexagonal(3, 5), [0, 0, 0], 7),
        (Lattice.rhombohedral(3, 5), [0, 0, 0], 7),
        (Lattice.rhombohedral(3, 5), [0.1, 0, 0], 3),
        (Lattice.rhombohedral(3, 5), [0.5, 0, 0], 3),
    ],
)
def test_get_image_sites_lattices(
    lattice: Lattice, site_coords: list[float], expected_images: int
) -> None:
    site = PeriodicSite("Si", site_coords, lattice)
    image_atoms = get_image_sites(site, lattice)
    assert len(image_atoms) == expected_images


@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize("is_image", [True, False])
@pytest.mark.parametrize(
    "hover_text",
    [*SiteCoords, lambda site: f"Custom: {site.species_string}"],
)
def test_draw_site(
    structures: list[Structure],
    mock_figure: Any,
    is_3d: bool,
    is_image: bool,
    hover_text: SiteCoords | Callable[[PeriodicSite], str],
) -> None:
    structure = structures[0]
    site = structure[0]
    coords = site.coords
    elem_colors = get_elem_colors(ElemColorScheme.jmol)
    atomic_radii = get_atomic_radii(None)

    draw_site(
        mock_figure,
        site,
        coords,
        0,
        "symbol",
        elem_colors,
        atomic_radii,
        atom_size=40,
        scale=1,
        site_kwargs={},
        is_3d=is_3d,
        is_image=is_image,
        hover_text=hover_text,
    )

    # Test with custom site labels
    custom_labels = {site.species_string: "Custom"}
    draw_site(
        mock_figure,
        site,
        coords,
        0,
        custom_labels,
        elem_colors,
        atomic_radii,
        atom_size=40,
        scale=1,
        site_kwargs={},
        is_3d=is_3d,
        is_image=is_image,
        hover_text=hover_text,
    )

    # check if the hover text is generated correctly
    if callable(hover_text):
        assert "Custom:" in mock_figure.last_trace_kwargs.get("hovertext", "")
    elif hover_text == SiteCoords.cartesian:
        assert "Coordinates (0, 0, 0)" in mock_figure.last_trace_kwargs["hovertext"]
    elif hover_text == SiteCoords.fractional:
        assert "Coordinates [0, 0, 0]" in mock_figure.last_trace_kwargs["hovertext"]
    elif hover_text == SiteCoords.cartesian_fractional:
        assert (
            "Coordinates (0, 0, 0) [0, 0, 0]"
            in mock_figure.last_trace_kwargs["hovertext"]
        )


@pytest.mark.parametrize(
    ("struct_key", "subplot_title", "expected_text"),
    [
        ("key", None, "key"),
        (1, None, "1. Si2 (spg="),  # Partial match due to dynamic spg number
        ("key", lambda _struct, key: f"Custom title for {key}", "Custom title for key"),
        (
            "key",
            lambda _struct, key: {"text": f"Custom for {key}", "font": {"size": 14}},
            "Custom for key",
        ),
    ],
)
def test_get_subplot_title(
    structures: list[Structure],
    struct_key: Any,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None,
    expected_text: str,
) -> None:
    structure = structures[0]
    title = get_subplot_title(structure, struct_key, 1, subplot_title)
    assert isinstance(title, dict)
    assert "text" in title
    assert expected_text in title["text"]

    if callable(subplot_title) and isinstance(
        subplot_title(structure, struct_key), dict
    ):
        assert "font" in title
        assert title["font"]["size"] == 14


def test_constants() -> None:
    assert isinstance(NO_SYM_MSG, str)
    assert NO_SYM_MSG.startswith("Symmetry could not be determined")
    assert isinstance(UNIT_CELL_EDGES, tuple)
    assert len(UNIT_CELL_EDGES) == 12
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in UNIT_CELL_EDGES)


@pytest.mark.parametrize(
    ("start", "vector", "is_3d", "arrow_kwargs", "expected_traces"),
    [
        # One for the line, one for the cone
        (
            [0, 0, 0],
            [1, 1, 1],
            True,
            {"color": "red", "width": 2, "arrow_head_length": 0.5},
            2,
        ),
        # One scatter trace for 2D
        ([0, 0], [1, 1], False, {"color": "blue", "width": 3}, 1),
        # One for the line, one for the cone
        ([1, 1, 1], [2, 2, 2], True, {"color": "green", "scale": 0.5}, 2),
        # One scatter trace for 2D
        ([1, 1], [2, 2], False, {"color": "yellow", "scale": 2}, 1),
    ],
)
def test_draw_vector(
    start: list[float],
    vector: list[float],
    is_3d: bool,
    arrow_kwargs: dict[str, Any],
    expected_traces: int,
) -> None:
    fig, start, vector = go.Figure(), np.array(start), np.array(vector)
    initial_trace_count = len(fig.data)
    draw_vector(fig, start, vector, is_3d=is_3d, arrow_kwargs=arrow_kwargs)
    assert len(fig.data) - initial_trace_count == expected_traces

    if is_3d:
        # Check 3D arrow properties
        line_trace = fig.data[-2]
        cone_trace = fig.data[-1]
        assert line_trace.mode == "lines"
        assert line_trace.line.color == arrow_kwargs["color"]
        assert line_trace.line.width == arrow_kwargs.get("width", 5)
        assert cone_trace.type == "cone"
        assert cone_trace.colorscale[0][1] == arrow_kwargs["color"]
        assert cone_trace.sizeref == arrow_kwargs.get("arrow_head_length", 0.8)
    else:
        # Check 2D arrow properties
        scatter_trace = fig.data[-1]
        assert scatter_trace.mode == "lines+markers"
        assert scatter_trace.marker.color == arrow_kwargs["color"]
        assert scatter_trace.line.width == arrow_kwargs.get("width", 5)

    # Check scaling
    scale = arrow_kwargs.get("scale", 1.0)
    end_point = start + vector * scale
    if is_3d:
        assert_allclose(fig.data[-2].x[1], end_point[0])
        assert_allclose(fig.data[-2].y[1], end_point[1])
        assert_allclose(fig.data[-2].z[1], end_point[2])
    else:
        assert_allclose(fig.data[-1].x[1], end_point[0])
        assert_allclose(fig.data[-1].y[1], end_point[1])


def test_draw_vector_default_values() -> None:
    fig = go.Figure()
    start = np.array([0, 0, 0])
    vector = np.array([1, 1, 1])
    draw_vector(fig, start, vector, is_3d=True)

    assert len(fig.data) == 2
    line_trace = fig.data[0]
    cone_trace = fig.data[1]

    assert line_trace.line.color == "white"
    assert line_trace.line.width == 5
    assert cone_trace.sizeref == 0.8
    assert_allclose(cone_trace.x, [1])
    assert_allclose(cone_trace.y, [1])
    assert_allclose(cone_trace.z, [1])


@pytest.fixture
def test_structures() -> list[Structure]:
    lattice = Lattice.cubic(5.0)
    struct1 = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct1.add_site_property("force", [[1, 1, 1], [-1, -1, -1]])
    struct1.add_site_property("charge", [1, -1])

    struct2 = Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct2.add_site_property("magmom", [5, -5])
    struct2.properties["energy"] = -10.0

    return [struct1, struct2]


@pytest.mark.parametrize(
    ("prop_keys", "expected_result"),
    [
        (["force", "magmom"], "force"),
        (["magmom", "force"], "magmom"),
        (["energy"], "energy"),
        (["non_existent"], None),
        ([], None),
    ],
)
def test_get_first_matching_site_prop(
    test_structures: list[Structure], prop_keys: list[str], expected_result: str | None
) -> None:
    assert get_first_matching_site_prop(test_structures, prop_keys) == expected_result


def test_get_first_matching_site_prop_with_filter(
    test_structures: list[Structure],
) -> None:
    def filter_positive(_prop: str, value: Any) -> bool:
        return isinstance(value, int | float) and value > 0

    assert (
        get_first_matching_site_prop(
            test_structures, ["charge", "magmom"], filter_callback=filter_positive
        )
        == "charge"
    )


def test_get_first_matching_site_prop_warning(test_structures: list[Structure]) -> None:
    with pytest.warns(UserWarning, match="None of prop_keys="):
        get_first_matching_site_prop(
            test_structures, ["non_existent"], warn_if_none=True
        )

    assert (
        get_first_matching_site_prop(
            test_structures, ["non_existent"], warn_if_none=False
        )
        is None
    )


def test_get_first_matching_site_prop_edge_cases() -> None:
    assert get_first_matching_site_prop([], ["force"]) is None

    lattice = Lattice.cubic(5.0)
    empty_struct = Structure(lattice, ["Si"], [[0, 0, 0]])
    assert get_first_matching_site_prop([empty_struct], ["force"]) is None

    multi_prop_struct = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    multi_prop_struct.add_site_property("force", [[1, 1, 1], [-1, -1, -1]])
    multi_prop_struct.add_site_property(
        "velocity", [[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]]
    )
    assert (
        get_first_matching_site_prop([multi_prop_struct], ["force", "velocity"])
        == "force"
    )

    def complex_filter(prop: str, value: Any) -> bool:
        if prop == "force":
            return all(abs(v) > 0.5 for v in value)
        if prop == "velocity":
            return all(abs(v) < 0.5 for v in value)
        return False

    assert (
        get_first_matching_site_prop(
            [multi_prop_struct], ["velocity", "force"], filter_callback=complex_filter
        )
        == "velocity"
    )


# Add this new test function for get_site_hover_text
@pytest.mark.parametrize(
    ("hover_text", "expected_output"),
    [
        (SiteCoords.cartesian, "(0, 0, 0)"),
        (SiteCoords.fractional, "[0, 0, 0]"),
        (SiteCoords.cartesian_fractional, "(0, 0, 0) [0, 0, 0]"),
        (lambda site: f"Custom: {site.species_string}", "Custom: Si"),
    ],
)
def test_get_site_hover_text(
    hover_text: SiteCoords | Callable[[PeriodicSite], str], expected_output: str
) -> None:
    lattice = Lattice.cubic(1.0)
    site = PeriodicSite("Si", [0, 0, 0], lattice)
    result = get_site_hover_text(site, hover_text, site.species)
    if isinstance(hover_text, str):
        expected_output = f"<b>Site: Si1</b><br>Coordinates {expected_output}"
    assert result == expected_output


def test_get_site_hover_text_invalid_template() -> None:
    lattice = Lattice.cubic(1.0)
    site = PeriodicSite("Si", [0, 0, 0], lattice)
    with pytest.raises(ValueError, match="Invalid hover_text="):
        get_site_hover_text(site, "invalid_template", site.species)  # type: ignore[arg-type]


@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize(
    "unit_cell_kwargs",
    [
        {},
        {"edge": {"color": "red", "width": 2, "dash": "solid"}},
        {"node": {"size": 5, "color": "blue"}},
        {
            "edge": {"color": "green", "width": 3},
            "node": {"size": 4, "color": "yellow"},
        },
    ],
)
def test_draw_unit_cell(
    structures: list[Structure], is_3d: bool, unit_cell_kwargs: dict[str, Any]
) -> None:
    structure = structures[0]  # Use the first structure from the fixture
    fig = go.Figure()

    draw_unit_cell(fig, structure, unit_cell_kwargs, is_3d=is_3d)

    # Check if the correct number of traces were added (12 edges + 8 corners = 20)
    assert len(fig.data) == 20
    trace_types = {*map(type, fig.data)}
    assert trace_types == {go.Scatter3d} if is_3d else {go.Scatter}
    trace_modes = [trace.mode for trace in fig.data]
    assert trace_modes.count("lines") == 12
    assert trace_modes.count("markers") == 8

    # Check if all traces are of the correct type
    expected_trace_type = go.Scatter3d if is_3d else go.Scatter
    assert all(isinstance(trace, expected_trace_type) for trace in fig.data)

    # Check if edge properties are applied correctly
    if "edge" in unit_cell_kwargs:
        edge_trace = fig.data[0]
        for key, value in unit_cell_kwargs["edge"].items():
            assert getattr(edge_trace.line, key) == value

    # Check if node properties are applied correctly
    if "node" in unit_cell_kwargs:
        node_trace = fig.data[12]  # First node trace
        for key, value in unit_cell_kwargs["node"].items():
            assert getattr(node_trace.marker, key) == value


def test_draw_unit_cell_hover_text(structures: list[Structure]) -> None:
    structure = structures[0]  # Use the first structure from the fixture
    fig = go.Figure()
    draw_unit_cell(fig, structure, {}, is_3d=True)

    # Check hover text for an edge
    edge_trace = fig.data[0]
    hover_text = edge_trace.hovertext[1]  # Middle point of the edge
    assert "Length:" in hover_text
    assert "Start:" in hover_text
    assert "End:" in hover_text

    # Check hover text for a corner
    corner_trace = fig.data[12]  # First corner trace
    hover_text = corner_trace.hovertext
    assert "α =" in hover_text  # noqa: RUF001
    assert "β =" in hover_text
    assert "γ =" in hover_text  # noqa: RUF001


@pytest.mark.parametrize(
    ("is_3d", "bond_kwargs", "visible_image_atoms"),
    [
        (True, None, None),
        (True, {"color": "blue", "width": 0.2}, {(1, 1, 1)}),
        (False, {"color": "red", "width": 2}, {(3.0, 3.0, 3.0)}),
        (
            True,
            {"color": "blue", "width": 3, "dash": "dot"},
            {(3.0, 3.0, 3.0), (-3.0, -3.0, -3.0)},
        ),
        (False, {"color": "green", "width": 1}, None),
    ],
)
def test_draw_bonds(
    is_3d: bool,
    bond_kwargs: dict[str, Any] | None,
    visible_image_atoms: set[tuple[float, float, float]] | None,
) -> None:
    # Create a simple structure
    lattice = Lattice.cubic(3.0)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Create a figure and a NearNeighbors object
    fig = go.Figure()
    nn = CrystalNN()

    # Draw bonds
    draw_bonds(
        fig=fig,
        structure=structure,
        nn=nn,
        is_3d=is_3d,
        bond_kwargs=bond_kwargs,
        visible_image_atoms=visible_image_atoms,
    )

    # Check if bonds were added
    assert len(fig.data) > 0

    # Check if all traces are of the correct type
    expected_trace_type = go.Scatter3d if is_3d else go.Scatter
    assert all(isinstance(trace, expected_trace_type) for trace in fig.data)

    # Check if custom bond properties were applied
    if bond_kwargs:
        for trace in fig.data:
            for key, value in bond_kwargs.items():
                assert getattr(trace.line, key) == value

    # Check if bonds to visible image atoms were added when applicable
    if visible_image_atoms:
        max_coords = max(
            max(trace.x + trace.y + (trace.z if is_3d else ())) for trace in fig.data
        )
        assert max_coords >= 1.5  # Should be greater than half the lattice parameter

    # Additional checks
    for trace in fig.data:
        assert trace.mode == "lines"
        assert trace.showlegend is False
        assert trace.hoverinfo == "skip"

    # Check the number of bonds (should be at least 1 for the central bond)
    assert len(fig.data) >= 1

    # If visible_image_atoms is provided, check for additional bonds
    if visible_image_atoms:
        assert len(fig.data) > 1  # Should have more than just the central bond
