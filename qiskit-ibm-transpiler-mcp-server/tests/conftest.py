"""Test configuration and fixtures for Qiskit IBM Transpiler MCP Server tests."""

import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_circuit_qasm():
    """Mock qasm circuit"""
    return "dummy_circuit_qasm"


@pytest.fixture
def mock_backend():
    """Mock backend name"""
    return "fake_backend"


@pytest.fixture
def get_backend_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def load_qasm_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def pass_manager_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def dumps_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_routing_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_clifford_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_linear_function_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_permutation_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_pauli_networks_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def mock_load_qasm_circuit_success(mocker):
    """Successful loading of QuantumCircuit object from QASM3.0 string"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.load_qasm_circuit")
    mock.return_value = {"status": "success", "circuit": "input_circuit"}
    return mock


@pytest.fixture
def mock_load_qasm_circuit_failure(mocker):
    """Failed loading of QuantumCircuit object from QASM3.0 string"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.load_qasm_circuit")
    mock.return_value = {
        "status": "error",
        "message": "Error in loading QuantumCircuit from QASM3.0",
    }
    return mock


@pytest.fixture
def mock_dumps_qasm_success(mocker):
    """Successful dumps methods for QASM3.0 string"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.dumps")
    mock.return_value = "optimized_circuit"
    return mock


@pytest.fixture
def mock_dumps_qasm_failure(mocker):
    """Failed dumps methods for QASM3.0 string"""
    mock = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.dumps",
        side_effect=Exception("QASM dumps failed"),
    )
    return mock


@pytest.fixture
def mock_get_backend_service_success(mocker):
    """Successful get_backend_service procedure"""
    mock = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.get_backend_service",
        new_callable=AsyncMock,
    )
    mock.return_value = {"backend": "mock_backend_object", "status": "success"}
    return mock


@pytest.fixture
def mock_get_backend_service_failure(mocker):
    """Failed get_backend_service procedure"""
    mock = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.get_backend_service",
        new_callable=AsyncMock,
    )
    mock.return_value = {"message": "get_backend failed", "status": "error"}
    return mock


@pytest.fixture
def mock_ai_routing_success(mocker):
    """Successful AIRouting procedure"""
    mock_class = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.AIRouting")
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_ai_routing_failure(mocker):
    """Failed AIRouting procedure"""
    mock_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIRouting",
        side_effect=Exception("AIRouting failed"),
    )
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    return mock_instance


@pytest.fixture
def mock_pass_manager_success(mocker):
    """Successful PassManager run procedure"""
    mock_pm_class = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.PassManager")
    mock_pm = MagicMock()
    mock_pm.run.return_value = "optimized_circuit"
    mock_pm_class.return_value = mock_pm
    return mock_pm


@pytest.fixture
def mock_pass_manager_failure(mocker):
    """Failed PassManager run procedure"""
    mock_pm_class = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.PassManager")
    mock_pm = MagicMock()
    mock_pm.run.side_effect = Exception("PassManager run failed")
    mock_pm_class.return_value = mock_pm
    return mock_pm


@pytest.fixture
def mock_ai_clifford_synthesis_success(mocker):
    """Successful AI Clifford synthesis procedure"""
    mock_clifford_synthesis_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AICliffordSynthesis"
    )
    mock_clifford_synthesis_instance = MagicMock()
    mock_clifford_synthesis_class.return_value = mock_clifford_synthesis_instance
    return mock_clifford_synthesis_class


@pytest.fixture
def mock_ai_clifford_synthesis_failure(mocker):
    """Failed AI Clifford synthesis procedure"""
    mock_clifford_synthesis_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AICliffordSynthesis",
        side_effect=Exception("AI Clifford synthesis failed"),
    )
    mock_clifford_synthesis_instance = MagicMock()
    mock_clifford_synthesis_class.return_value = mock_clifford_synthesis_instance
    return mock_clifford_synthesis_instance


@pytest.fixture
def mock_ai_linear_function_synthesis_success(mocker):
    """Successful AI Linear Function synthesis procedure"""
    mock_linear_function_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AILinearFunctionSynthesis"
    )
    mock_linear_function_instance = MagicMock()
    mock_linear_function_class.return_value = mock_linear_function_instance
    return mock_linear_function_class


@pytest.fixture
def mock_ai_linear_function_synthesis_failure(mocker):
    """Failed AI Linear Function synthesis procedure"""
    mock_linear_function_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AILinearFunctionSynthesis",
        side_effect=Exception("AI Linear Function synthesis failed"),
    )
    mock_linear_function_instance = MagicMock()
    mock_linear_function_class.return_value = mock_linear_function_instance
    return mock_linear_function_instance


@pytest.fixture
def mock_ai_permutation_synthesis_success(mocker):
    """Successful AI Permutation synthesis procedure"""
    mock_permutation_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIPermutationSynthesis"
    )
    mock_permutation_instance = MagicMock()
    mock_permutation_class.return_value = mock_permutation_instance
    return mock_permutation_class


@pytest.fixture
def mock_ai_permutation_synthesis_failure(mocker):
    """Failed AI Permutation synthesis procedure"""
    mock_permutation_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIPermutationSynthesis",
        side_effect=Exception("Permutation synthesis failed"),
    )
    mock_permutation_instance = MagicMock()
    mock_permutation_class.return_value = mock_permutation_instance
    return mock_permutation_instance


@pytest.fixture
def mock_ai_pauli_network_synthesis_success(mocker):
    """Successful AI Pauli Networks synthesis procedure"""
    mock_pauli_network_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIPauliNetworkSynthesis"
    )
    mock_pauli_network_instance = MagicMock()
    mock_pauli_network_class.return_value = mock_pauli_network_instance
    return mock_pauli_network_class


@pytest.fixture
def mock_ai_pauli_network_synthesis_failure(mocker):
    """Failed AI Pauli Networks synthesis procedure"""
    mock_pauli_network_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIPauliNetworkSynthesis",
        side_effect=Exception("Pauli Networks synthesis failed"),
    )
    mock_pauli_network_instance = MagicMock()
    mock_pauli_network_class.return_value = mock_pauli_network_instance
    return mock_pauli_network_instance
