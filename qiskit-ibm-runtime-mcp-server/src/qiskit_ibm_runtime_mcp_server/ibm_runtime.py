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

"""Core IBM Runtime functions for the MCP server."""

import logging
import os
from typing import Any, Dict, Optional

from qiskit_ibm_runtime.accounts import ChannelType
from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import-untyped]
from qiskit_ibm_runtime import EstimatorV2


def least_busy(backends):
    """Find the least busy backend from a list of backends."""
    if not backends:
        return None

    operational_backends = [
        b for b in backends if hasattr(b, "status") and b.status().operational
    ]
    if not operational_backends:
        return None

    return min(operational_backends, key=lambda b: b.status().pending_jobs)


def get_token_from_env() -> Optional[str]:
    """
    Get IBM Quantum token from environment variables.

    Returns:
        Token string if found in environment, None otherwise
    """
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if (
        token
        and token.strip()
        and token.strip() not in ["<PASSWORD>", "<TOKEN>", "YOUR_TOKEN_HERE"]
    ):
        return token.strip()
    return None


logger = logging.getLogger(__name__)

# Global service instances
service: Optional[QiskitRuntimeService] = None
estimator_service: Optional[EstimatorV2] = None


def initialize_service(
    token: Optional[str] = None, channel: str = "ibm_quantum_platform"
) -> QiskitRuntimeService:
    """
    Initialize the Qiskit IBM Runtime service.

    Args:
        token: IBM Quantum API token (optional if saved)
        channel: Service channel ('ibm_quantum_platform')

    Returns:
        QiskitRuntimeService: Initialized service instance
    """
    global service

    try:
        # First, try to initialize from saved credentials (unless a new token is explicitly provided)
        if not token:
            try:
                service = QiskitRuntimeService(channel=channel)
                logger.info(
                    f"Successfully initialized IBM Runtime service from saved credentials on channel: {channel}"
                )
                return service
            except Exception as e:
                logger.info(f"No saved credentials found or invalid: {e}")
                raise ValueError(
                    "No IBM Quantum token provided and no saved credentials available"
                ) from e

        # If a token is provided, validate it's not a placeholder before saving
        if token and token.strip():
            # Check for common placeholder patterns
            if token.strip() in ["<PASSWORD>", "<TOKEN>", "YOUR_TOKEN_HERE", "xxx"]:
                raise ValueError(
                    f"Invalid token: '{token.strip()}' appears to be a placeholder value"
                )

            # Save account with provided token
            try:
                QiskitRuntimeService.save_account(
                    channel=channel, token=token.strip(), overwrite=True
                )
                logger.info(f"Saved IBM Quantum account for channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to save account: {e}")
                raise ValueError("Invalid token or channel") from e

            # Initialize service with the new token
            try:
                service = QiskitRuntimeService(channel=channel)
                logger.info(
                    f"Successfully initialized IBM Runtime service on channel: {channel}"
                )
                return service
            except Exception as e:
                logger.error(f"Failed to initialize IBM Runtime service: {e}")
                raise

    except Exception as e:
        if not isinstance(e, ValueError):
            logger.error(f"Failed to initialize IBM Runtime service: {e}")
        raise

def initialize_estimator(
    mode = None,
    options = None
) -> EstimatorV2:
    """Initialize the EstimatorV2 instance.

    This helper creates an :class:`qiskit_ibm_runtime.EstimatorV2` that
    interacts with Qiskit Runtime Estimator primitive service.
        Qiskit Runtime Estimator primitive service estimates expectation values of quantum circuits and
        observables.

    Args:
            mode: The execution mode used to make the primitive query. It can be:

                * A :class:`Backend` if you are using job mode.
                * A :class:`Session` if you are using session execution mode.
                * A :class:`Batch` if you are using batch execution mode.

                Refer to the
                `Qiskit Runtime documentation
                <https://quantum.cloud.ibm.com/docs/guides/execution-modes>`_
                for more information about the ``Execution modes``.

            options: Estimator options, see :class:`EstimatorOptions` for detailed description.

    Returns:
        EstimatorV2: The initialized estimator.
    """
    global estimator_service
    try:
        # Create the estimator that talks to the quantum back‑ends.
        estimator_service = EstimatorV2(mode=mode, options=options)
        logger.info("Successfully initialized EstimatorV2.")
        return estimator_service
    except Exception as e:
        logger.error(f"Failed to initialize EstimatorV2: {e}")
        raise

async def setup_ibm_quantum_account(
    token: Optional[str] = None, channel: str = "ibm_quantum_platform"
) -> Dict[str, Any]:
    """
    Set up IBM Quantum account with credentials.

    Args:
        token: IBM Quantum API token (optional - will try environment or saved credentials)
        channel: Service channel ('ibm_quantum_platform')

    Returns:
        Setup status and information
    """
    # Try to get token from environment if not provided
    if not token or not token.strip():
        env_token = get_token_from_env()
        if env_token:
            logger.info("Using token from IBM_QUANTUM_TOKEN environment variable")
            token = env_token
        else:
            # Try to use saved credentials
            logger.info("No token provided, attempting to use saved credentials")
            token = None

    if channel not in ["ibm_quantum_platform"]:
        return {
            "status": "error",
            "message": "Channel must be 'ibm_quantum_platform'",
        }

    try:
        service_instance = initialize_service(token.strip() if token else None, channel)

        # Get backend count for response
        try:
            backends = service_instance.backends()
            backend_count = len(backends)
        except Exception:
            backend_count = 0

        return {
            "status": "success",
            "message": f"IBM Quantum account set up successfully for channel: {channel}",
            "channel": service_instance._channel,
            "available_backends": backend_count,
        }
    except Exception as e:
        logger.error(f"Failed to set up IBM Quantum account: {e}")
        return {"status": "error", "message": f"Failed to set up account: {str(e)}"}


async def list_backends() -> Dict[str, Any]:
    """
    List available IBM Quantum backends.

    Returns:
        List of backends with their properties
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backends = service.backends()
        backend_list = []

        for backend in backends:
            try:
                status = backend.status()
                backend_info = {
                    "name": backend.name,
                    "num_qubits": getattr(backend, "num_qubits", 0),
                    "simulator": getattr(backend, "simulator", False),
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "status_msg": status.status_msg,
                }
                backend_list.append(backend_info)
            except Exception as be:
                logger.warning(f"Failed to get status for backend {backend.name}: {be}")
                backend_info = {
                    "name": backend.name,
                    "num_qubits": getattr(backend, "num_qubits", 0),
                    "simulator": getattr(backend, "simulator", False),
                    "operational": False,
                    "pending_jobs": 0,
                    "status_msg": "Status unavailable",
                }
                backend_list.append(backend_info)

        return {
            "status": "success",
            "backends": backend_list,
            "total_backends": len(backend_list),
        }

    except Exception as e:
        logger.error(f"Failed to list backends: {e}")
        return {"status": "error", "message": f"Failed to list backends: {str(e)}"}


async def least_busy_backend() -> Dict[str, Any]:
    """
    Find the least busy operational backend.

    Returns:
        Information about the least busy backend
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backends = service.backends(operational=True, simulator=False)

        if not backends:
            return {
                "status": "error",
                "message": "No operational quantum backends available",
            }

        backend = least_busy(backends)
        status = backend.status()

        return {
            "status": "success",
            "backend_name": backend.name,
            "num_qubits": getattr(backend, "num_qubits", 0),
            "pending_jobs": status.pending_jobs,
            "operational": status.operational,
            "status_msg": status.status_msg,
        }

    except Exception as e:
        logger.error(f"Failed to find least busy backend: {e}")
        return {
            "status": "error",
            "message": f"Failed to find least busy backend: {str(e)}",
        }


async def get_backend_properties(backend_name: str) -> Dict[str, Any]:
    """
    Get detailed properties of a specific backend.

    Args:
        backend_name: Name of the backend

    Returns:
        Backend properties and capabilities
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backend = service.backend(backend_name)
        status = backend.status()

        # Get configuration
        try:
            config = backend.configuration()
            basis_gates = getattr(config, "basis_gates", [])
            coupling_map = getattr(config, "coupling_map", [])
            max_shots = getattr(config, "max_shots", 0)
            max_experiments = getattr(config, "max_experiments", 0)
        except Exception:
            basis_gates = []
            coupling_map = []
            max_shots = 0
            max_experiments = 0

        properties = {
            "status": "success",
            "backend_name": backend.name,
            "num_qubits": getattr(backend, "num_qubits", 0),
            "simulator": getattr(backend, "simulator", False),
            "operational": status.operational,
            "pending_jobs": status.pending_jobs,
            "status_msg": status.status_msg,
            "basis_gates": basis_gates,
            "coupling_map": coupling_map,
            "max_shots": max_shots,
            "max_experiments": max_experiments,
        }

        return properties

    except Exception as e:
        logger.error(f"Failed to get backend properties: {e}")
        return {
            "status": "error",
            "message": f"Failed to get backend properties: {str(e)}",
        }


async def list_my_jobs(limit: int = 10) -> Dict[str, Any]:
    """
    List user's recent jobs.

    Args:
        limit: Maximum number of jobs to retrieve

    Returns:
        List of jobs with their information
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        jobs = service.jobs(limit=limit)
        job_list = []

        for job in jobs:
            try:
                job_info = {
                    "job_id": job.job_id(),
                    "status": job.status(),
                    "creation_date": getattr(job, "creation_date", "Unknown"),
                    "backend": job.backend().name if job.backend() else "Unknown",
                    "tags": getattr(job, "tags", []),
                    "error_message": job.error_message()
                    if hasattr(job, "error_message")
                    else None,
                }
                job_list.append(job_info)
            except Exception as je:
                logger.warning(f"Failed to get info for job: {je}")
                continue

        return {"status": "success", "jobs": job_list, "total_jobs": len(job_list)}

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return {"status": "error", "message": f"Failed to list jobs: {str(e)}"}


async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get status of a specific job.

    Args:
        job_id: ID of the job

    Returns:
        Job status information
    """
    global service

    try:
        if service is None:
            return {
                "status": "error",
                "message": "Failed to get job status: service not initialized",
            }

        job = service.job(job_id)

        job_info = {
            "status": "success",
            "job_id": job.job_id(),
            "job_status": job.status(),
            "creation_date": getattr(job, "creation_date", "Unknown"),
            "backend": job.backend().name if job.backend() else "Unknown",
            "tags": getattr(job, "tags", []),
            "error_message": job.error_message()
            if hasattr(job, "error_message")
            else None,
        }

        return job_info

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return {"status": "error", "message": f"Failed to get job status: {str(e)}"}


async def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a specific job.

    Args:
        job_id: ID of the job to cancel

    Returns:
        Cancellation status
    """
    global service

    try:
        if service is None:
            return {
                "status": "error",
                "message": "Failed to cancel job: service not initialized",
            }

        job = service.job(job_id)
        job.cancel()

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Job cancellation requested",
        }
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        return {"status": "error", "message": f"Failed to cancel job: {str(e)}"}


async def get_service_status() -> str:
    """
    Get current IBM Quantum service status.

    Returns:
        Service connection status and basic information
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        # Test connectivity by listing backends
        backends = service.backends()
        backend_count = len(backends)

        status_info = {
            "connected": True,
            "channel": service._channel,
            "available_backends": backend_count,
            "service": "IBM Quantum",
        }

        return f"IBM Quantum Service Status: {status_info}"

    except Exception as e:
        logger.error(f"Failed to check service status: {e}")
        status_info = {"connected": False, "error": str(e), "service": "IBM Quantum"}
        return f"IBM Quantum Service Status: {status_info}"

async def delete_saved_account(filename: str = None, name: str = None, channel: ChannelType = None,) -> str:
    """Delete a saved account.
    Args:
        filename: Name of file from which to delete the account.
        name: Name of the saved account to delete.
        channel: Channel type of the default account to delete.
            Ignored if account name is provided.

    Returns:
    Information about if an account was deleted.
    """
    global service

    try:
        if service is None:
            service = initialize_service()
        service.delete_account(filename, name, channel)

        delete_info = {
            "deleted": True,
            "channel": channel,
            "name":name
        }
        return f"Account successfully deleted {delete_info}"
    except Exception as e:
        logger.error(f"Failed to delete account: {e}")
        delete_info = {"deleted": False, "filename": filename, "name": name, "channel": channel, "error": str(e)}
        return f"Error deleting account: {delete_info}"
    
async def list_saved_account(default: str = None, channel: ChannelType = None, filename: str = None, name: str = None) -> str:
    """List the accounts saved on disk.
    Args:
        default: If set to True, only default accounts are returned.
        channel: Channel type.``ibm_cloud`` or ``ibm_quantum_platform``.
        filename: Name of file whose accounts are returned.
        name: If set, only accounts with the given name are returned.

    Returns:
        List of all saved accounts.
    """
    global service

    try:
        if service is None:
            service = initialize_service()
        accounts_list = service.saved_accounts(filename, name, channel)
        return f"Account list: {accounts_list}"
    except Exception as e:
        logger.error(f"Failed to collect accounts: {e}")
        collect_info = {"collect": False, "filename": filename, "name": name, "channel": channel, "error": str(e)}
        return f"Error collecting accounts: {collect_info}"
    
async def active_account_info() -> str:
    """
    Return the IBM Quantum account currently in use for the session.
        Returns:
            A dictionary with information about the account currently in the session.
        """
    global service

    try:
        if service is None:
            service = initialize_service()
        active_account_info = service.active_account()
        return f"Active account info: {active_account_info}"
    except Exception as e:
        logger.error(f"Failed to collect active account: {e}")
        active_account_info = {"collect": False, "error": str(e)}
        return f"Error collecting active account: {active_account_info}"

async def active_instance_info() -> str:
    """Return the crn of the current active instance."""
    global service

    try:
        if service is None:
            service = initialize_service()
        active_instance_info = service.active_instance()
        return f"Active instance crn: {active_instance_info}"
    except Exception as e:
        logger.error(f"Failed to collect instance crn: {e}")
        active_instance_info = {"collect": False, "error": str(e)}
        return f"Error collecting active instance crn: {active_instance_info}"

async def available_instances() -> str:
    """Return a list that contains a series of dictionaries with the
            following instance identifiers per instance: "crn", "plan", "name".

        Returns:
            Instances available for the active account.
        """
    global service

    try:
        if service is None:
            service = initialize_service()
        available_instances_info = service.instances()
        return f"Available instances for the active account: {available_instances_info}"
    except Exception as e:
        logger.error(f"Failed to collect available instances for the active account: {e}")
        available_instances_info = {"collect": False, "error": str(e)}
        return f"Error collecting available instances for the active account: {available_instances_info}"
    
async def usage_info() -> str:
    """Return usage information for the current active instance.

        Returns:
            Instance usage details.
        """
    global service

    try:
        if service is None:
            service = initialize_service()
        usage_info = service.usage()
        return f"Usage information: {usage_info}"
    except Exception as e:
        logger.error(f"Failed to collect usage information: {e}")
        usage_info = {"collect": False, "error": str(e)}
        return f"Error collecting usage information: {usage_info}"

async def estimator_run(pubs = None, precision = None) -> str:
    """Submit a request to the estimator primitive.

        Args:
            pubs: An iterable of pub-like (primitive unified bloc) objects, such as
                tuples ``(circuit, observables)`` or ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                run Estimator Pub that does not specify its own precision. If None
                the estimator's default precision value will be used.

        Returns:
            Submitted job.

        Raises:
            ValueError: if precision value is not strictly greater than 0.
        """
    global estimator_service

    try:
        if estimator_service is None:
            estimator_service = initialize_estimator()
        job = estimator_service.run(pubs, precision)
        
        return f"Estimator job submitted: {job}"
    except Exception as e:
        logger.error(f"Failed run the estimator: {e}")
        job_info = {"run": False, "error": str(e)}
        return f"Error running the estimator: {job_info}"

# Assisted by watsonx Code Assistant
