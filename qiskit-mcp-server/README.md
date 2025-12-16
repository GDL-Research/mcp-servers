# Qiskit MCP Server - Shared Utilities

Shared utilities for Qiskit MCP servers, including circuit serialization (QPY/QASM3) and async helpers.

## Installation

```bash
pip install qiskit-mcp-server
```

## Features

### Circuit Serialization

Support for both QASM 3.0 (text) and QPY (binary, base64-encoded) circuit formats:

```python
from qiskit import QuantumCircuit
from qiskit_mcp_server import load_circuit, dump_circuit

# Create a circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Serialize to QASM3 (default)
qasm_str = dump_circuit(qc, circuit_format="qasm3")

# Serialize to QPY (base64-encoded, preserves full circuit fidelity)
qpy_str = dump_circuit(qc, circuit_format="qpy")

# Load from either format
result = load_circuit(qasm_str, circuit_format="qasm3")
result = load_circuit(qpy_str, circuit_format="qpy")
circuit = result["circuit"]
```

### Async Utilities

The `with_sync` decorator enables async functions to be called synchronously:

```python
from qiskit_mcp_server import with_sync

@with_sync
async def my_async_function(arg: str) -> dict:
    return {"result": arg}

# Async call
result = await my_async_function("hello")

# Sync call (useful in Jupyter notebooks, DSPy, etc.)
result = my_async_function.sync("hello")
```

## API Reference

### Circuit Serialization Functions

- `load_circuit(circuit_data, circuit_format="qasm3")` - Load circuit from QASM3 or QPY
- `dump_circuit(circuit, circuit_format="qasm3")` - Serialize circuit to QASM3 or QPY
- `load_qasm_circuit(qasm_string)` - Load circuit from QASM 3.0 string
- `load_qpy_circuit(qpy_b64)` - Load circuit from base64-encoded QPY
- `dump_qasm_circuit(circuit)` - Serialize circuit to QASM 3.0 string
- `dump_qpy_circuit(circuit)` - Serialize circuit to base64-encoded QPY

### Type Aliases

- `CircuitFormat` - Literal type for "qasm3" or "qpy"
