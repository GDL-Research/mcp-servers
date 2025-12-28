# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for coupling_maps.py."""

import pytest

from qiskit_gym_mcp_server.coupling_maps import (
    HARDWARE_PRESETS,
    create_coupling_map_from_edges,
    create_coupling_map_from_preset,
    create_custom_coupling_map,
    extract_subtopologies,
    extract_unique_subtopologies,
    generate_coupling_map_edges,
    get_coupling_map_presets,
    list_subtopology_shapes,
)


class TestHardwarePresets:
    """Tests for hardware presets."""

    def test_presets_exist(self):
        """Test that expected presets are defined."""
        assert "ibm_heron_r1" in HARDWARE_PRESETS
        assert "ibm_heron_r2" in HARDWARE_PRESETS
        assert "ibm_nighthawk" in HARDWARE_PRESETS
        assert "grid_3x3" in HARDWARE_PRESETS
        assert "linear_5" in HARDWARE_PRESETS

    def test_nighthawk_specs(self):
        """Test IBM Nighthawk has correct specifications."""
        nighthawk = HARDWARE_PRESETS["ibm_nighthawk"]
        assert nighthawk["num_qubits"] == 120
        assert nighthawk["topology"] == "grid"
        assert nighthawk["rows"] == 10
        assert nighthawk["cols"] == 12

    def test_heron_specs(self):
        """Test IBM Heron has correct specifications."""
        heron = HARDWARE_PRESETS["ibm_heron_r1"]
        assert heron["num_qubits"] == 133
        assert heron["topology"] == "heavy_hex"


class TestCouplingMapGeneration:
    """Tests for coupling map generation."""

    def test_generate_grid_edges(self):
        """Test grid coupling map generation."""
        edges = generate_coupling_map_edges("grid", rows=2, cols=2)
        # 2x2 grid should have 4 edges (bidirectional = 8)
        assert len(edges) == 8

    def test_generate_line_edges(self):
        """Test line coupling map generation."""
        edges = generate_coupling_map_edges("line", num_qubits=5)
        # 5-qubit line should have 4 edges (bidirectional = 8)
        assert len(edges) == 8

    def test_create_from_preset(self):
        """Test creating coupling map from preset."""
        _cmap, edges, num_qubits = create_coupling_map_from_preset("grid_3x3")
        assert num_qubits == 9
        assert len(edges) > 0

    def test_create_from_edges(self, sample_coupling_map_linear):
        """Test creating coupling map from custom edges."""
        _cmap, _edges, num_qubits = create_coupling_map_from_edges(
            sample_coupling_map_linear, bidirectional=False
        )
        assert num_qubits == 5

    def test_invalid_preset_raises(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_coupling_map_from_preset("invalid_preset")


class TestSubtopologyExtraction:
    """Tests for subtopology extraction."""

    def test_extract_subtopologies_from_grid(self):
        """Test extracting subtopologies from grid preset."""
        subtopologies = extract_unique_subtopologies("grid_3x3", num_qubits_target=4)
        assert len(subtopologies) > 0
        for sub in subtopologies:
            assert sub["num_qubits"] == 4
            assert "edges" in sub
            assert "shape" in sub

    def test_extract_subtopologies_too_many_qubits(self):
        """Test error when requesting more qubits than source has."""
        with pytest.raises(ValueError, match="source only has"):
            extract_unique_subtopologies("grid_3x3", num_qubits_target=100)

    @pytest.mark.asyncio
    async def test_extract_subtopologies_async(self):
        """Test async extract_subtopologies function."""
        result = await extract_subtopologies(preset="grid_3x3", num_qubits=3)
        assert result["status"] == "success"
        assert result["num_qubits"] == 3
        assert len(result["subtopologies"]) > 0


class TestAsyncFunctions:
    """Tests for async coupling map functions."""

    @pytest.mark.asyncio
    async def test_get_coupling_map_presets(self):
        """Test getting coupling map presets."""
        result = await get_coupling_map_presets()
        assert result["status"] == "success"
        assert "presets" in result
        assert "ibm_nighthawk" in result["presets"]

    @pytest.mark.asyncio
    async def test_create_custom_coupling_map_from_edges(self, sample_coupling_map_linear):
        """Test creating custom coupling map from edges."""
        result = await create_custom_coupling_map(edges=sample_coupling_map_linear)
        assert result["status"] == "success"
        assert result["num_qubits"] == 5

    @pytest.mark.asyncio
    async def test_create_custom_coupling_map_from_topology(self):
        """Test creating custom coupling map from topology."""
        result = await create_custom_coupling_map(topology="grid", rows=3, cols=3)
        assert result["status"] == "success"
        assert result["num_qubits"] == 9

    @pytest.mark.asyncio
    async def test_create_custom_coupling_map_error_both_params(self, sample_coupling_map_linear):
        """Test error when both edges and topology provided."""
        result = await create_custom_coupling_map(
            edges=sample_coupling_map_linear,
            topology="grid",
            rows=2,
            cols=2,
        )
        assert result["status"] == "error"
        assert "not both" in result["message"]

    @pytest.mark.asyncio
    async def test_list_subtopology_shapes(self):
        """Test listing subtopology shapes."""
        result = await list_subtopology_shapes(preset="grid_3x3", num_qubits=3)
        assert result["status"] == "success"
        assert "shape_counts" in result
        assert result["total_subtopologies"] > 0
