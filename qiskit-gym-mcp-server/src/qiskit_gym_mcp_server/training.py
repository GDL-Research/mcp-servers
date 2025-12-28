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

"""Training session management for qiskit-gym MCP server.

This module provides functions to:
- Start RL training sessions with configurable algorithms and policies
- Monitor training progress and metrics
- Stop running training sessions
- Batch train across multiple environments/topologies
"""

import logging
from pathlib import Path
from typing import Any, Literal

from qiskit_gym_mcp_server.constants import (
    QISKIT_GYM_MAX_ITERATIONS,
    QISKIT_GYM_TENSORBOARD_DIR,
)
from qiskit_gym_mcp_server.state import GymStateProvider
from qiskit_gym_mcp_server.utils import with_sync


logger = logging.getLogger(__name__)


def _get_rl_config(algorithm: str) -> Any:
    """Get the configuration class for an RL algorithm.

    Args:
        algorithm: Algorithm name ("ppo" or "alphazero")

    Returns:
        Config class instance
    """
    from qiskit_gym.rl import AlphaZeroConfig, PPOConfig

    if algorithm == "ppo":
        return PPOConfig()
    elif algorithm == "alphazero":
        return AlphaZeroConfig()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: ppo, alphazero")


def _get_policy_config(policy: str) -> Any:
    """Get the policy configuration class.

    Args:
        policy: Policy name ("basic" or "conv1d")

    Returns:
        Policy config class instance
    """
    from qiskit_gym.rl import BasicPolicyConfig, Conv1dPolicyConfig

    if policy == "basic":
        return BasicPolicyConfig()
    elif policy == "conv1d":
        return Conv1dPolicyConfig()
    else:
        raise ValueError(f"Unknown policy: {policy}. Supported: basic, conv1d")


def _ensure_tensorboard_dir(experiment_name: str | None) -> str | None:
    """Ensure TensorBoard directory exists and return path.

    Args:
        experiment_name: Name for the experiment

    Returns:
        Path to TensorBoard log directory, or None if not enabled
    """
    if experiment_name is None:
        return None

    tb_dir = Path(QISKIT_GYM_TENSORBOARD_DIR).expanduser()
    tb_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = tb_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    return str(experiment_dir)


# ============================================================================
# Training Functions
# ============================================================================


@with_sync
async def start_training(
    env_id: str,
    algorithm: Literal["ppo", "alphazero"] = "ppo",
    policy: Literal["basic", "conv1d"] = "basic",
    num_iterations: int = 100,
    tensorboard_experiment: str | None = None,
) -> dict[str, Any]:
    """Start training an RL agent on an environment.

    This initiates a training session that learns to synthesize optimal circuits.
    Training runs synchronously - the call will return when training completes.

    Args:
        env_id: Environment ID from create_*_env tools
        algorithm: RL algorithm to use:
            - "ppo": Proximal Policy Optimization (recommended for most cases)
            - "alphazero": AlphaZero-style MCTS (better for complex problems, slower)
        policy: Neural network policy architecture:
            - "basic": Simple feedforward network (faster, good for small problems)
            - "conv1d": 1D convolutional network (better for larger problems)
        num_iterations: Number of training iterations (default: 100)
        tensorboard_experiment: Name for TensorBoard experiment logging (optional)

    Returns:
        Dict with session_id, final status, and training metrics
    """
    try:
        from qiskit_gym.rl import RLSynthesis

        # Validate iteration count
        if num_iterations > QISKIT_GYM_MAX_ITERATIONS:
            return {
                "status": "error",
                "message": f"num_iterations ({num_iterations}) exceeds maximum ({QISKIT_GYM_MAX_ITERATIONS})",
            }

        if num_iterations < 1:
            return {
                "status": "error",
                "message": "num_iterations must be at least 1",
            }

        # Get environment
        state = GymStateProvider()
        env = state.get_environment(env_id)
        if env is None:
            return {
                "status": "error",
                "message": f"Environment '{env_id}' not found. Use list_environments to see available.",
            }

        # Set up TensorBoard path
        tb_path = _ensure_tensorboard_dir(tensorboard_experiment)

        # Create training session
        session_id = state.create_training_session(
            env_id=env_id,
            algorithm=algorithm,
            policy=policy,
            total_iterations=num_iterations,
            tensorboard_path=tb_path,
        )

        # Get configs
        rl_config = _get_rl_config(algorithm)
        policy_config = _get_policy_config(policy)

        # Create RLSynthesis instance
        rls = RLSynthesis(env.gym_instance, rl_config, policy_config)

        # Store RLS instance in session
        state.set_training_rls_instance(session_id, rls)
        state.set_training_status(session_id, "running")

        # Run training
        logger.info(f"Starting training session {session_id} with {num_iterations} iterations")
        try:
            if tb_path:
                rls.learn(num_iterations=num_iterations, tb_path=tb_path)
            else:
                rls.learn(num_iterations=num_iterations)

            # Training completed
            state.set_training_status(session_id, "completed")
            state.update_training_progress(session_id, num_iterations)

            # Register as a model
            model_id = state.register_model(
                model_name=f"trained_{env.env_type}_{session_id}",
                env_type=env.env_type,
                coupling_map_edges=env.coupling_map_edges,
                num_qubits=env.num_qubits,
                rls_instance=rls,
                from_session_id=session_id,
            )

            return {
                "status": "success",
                "session_id": session_id,
                "model_id": model_id,
                "env_id": env_id,
                "algorithm": algorithm,
                "policy": policy,
                "iterations_completed": num_iterations,
                "tensorboard_path": tb_path,
                "message": "Training completed successfully",
                "next_steps": [
                    f"Use save_model with session_id='{session_id}' to persist the model",
                    f"Use synthesize_{env.env_type} with model_id='{model_id}' to generate circuits",
                ],
            }

        except Exception as train_error:
            state.set_training_status(session_id, "error", str(train_error))
            raise

    except ImportError as e:
        logger.error(f"qiskit-gym not installed: {e}")
        return {
            "status": "error",
            "message": "qiskit-gym package not installed. Install with: pip install qiskit-gym",
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def get_training_status(session_id: str) -> dict[str, Any]:
    """Get the status and metrics of a training session.

    Args:
        session_id: Training session ID

    Returns:
        Dict with session status, progress, and metrics
    """
    state = GymStateProvider()
    session = state.get_training_session(session_id)

    if session is None:
        return {
            "status": "error",
            "message": f"Training session '{session_id}' not found",
        }

    return {
        "status": "success",
        "session_id": session.session_id,
        "env_id": session.env_id,
        "algorithm": session.algorithm,
        "policy": session.policy,
        "training_status": session.status,
        "progress": session.progress,
        "total_iterations": session.total_iterations,
        "progress_percent": round(100 * session.progress / session.total_iterations, 1)
        if session.total_iterations > 0
        else 0,
        "metrics": session.metrics,
        "tensorboard_path": session.tensorboard_path,
        "error_message": session.error_message,
    }


@with_sync
async def stop_training(session_id: str) -> dict[str, Any]:
    """Stop a running training session.

    Note: This marks the session as stopped but cannot interrupt
    an in-progress training iteration.

    Args:
        session_id: Training session ID to stop

    Returns:
        Dict with stop status
    """
    state = GymStateProvider()
    session = state.get_training_session(session_id)

    if session is None:
        return {
            "status": "error",
            "message": f"Training session '{session_id}' not found",
        }

    if session.status == "completed":
        return {
            "status": "error",
            "message": f"Training session '{session_id}' already completed",
        }

    if session.status == "stopped":
        return {
            "status": "error",
            "message": f"Training session '{session_id}' already stopped",
        }

    state.set_training_status(session_id, "stopped")

    return {
        "status": "success",
        "session_id": session_id,
        "message": "Training session marked as stopped",
        "progress_at_stop": session.progress,
    }


@with_sync
async def list_training_sessions() -> dict[str, Any]:
    """List all training sessions.

    Returns:
        Dict with list of training sessions
    """
    state = GymStateProvider()
    sessions = state.list_training_sessions()

    return {
        "status": "success",
        "sessions": sessions,
        "total": len(sessions),
    }


# ============================================================================
# Batch Training
# ============================================================================


@with_sync
async def batch_train_environments(
    env_ids: list[str],
    algorithm: Literal["ppo", "alphazero"] = "ppo",
    policy: Literal["basic", "conv1d"] = "basic",
    num_iterations: int = 100,
    tensorboard_prefix: str | None = None,
) -> dict[str, Any]:
    """Train multiple environments in sequence.

    Useful for training models across multiple topologies or subtopologies.

    Args:
        env_ids: List of environment IDs to train
        algorithm: RL algorithm to use
        policy: Neural network policy architecture
        num_iterations: Number of iterations per environment
        tensorboard_prefix: Prefix for TensorBoard experiment names

    Returns:
        Dict with results for each environment
    """
    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0

    for i, env_id in enumerate(env_ids):
        logger.info(f"Batch training {i + 1}/{len(env_ids)}: {env_id}")

        # Generate TensorBoard experiment name
        tb_name = None
        if tensorboard_prefix:
            tb_name = f"{tensorboard_prefix}_{env_id}"

        # Train this environment
        result = await start_training(
            env_id=env_id,
            algorithm=algorithm,
            policy=policy,
            num_iterations=num_iterations,
            tensorboard_experiment=tb_name,
        )

        result["env_id"] = env_id
        results.append(result)

        if result["status"] == "success":
            successful += 1
        else:
            failed += 1

    return {
        "status": "success" if failed == 0 else "partial",
        "total_environments": len(env_ids),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


# ============================================================================
# Training Configuration
# ============================================================================


@with_sync
async def get_available_algorithms() -> dict[str, Any]:
    """Get information about available RL algorithms.

    Returns:
        Dict with algorithm descriptions and recommendations
    """
    return {
        "status": "success",
        "algorithms": {
            "ppo": {
                "name": "Proximal Policy Optimization",
                "description": "Stable, sample-efficient policy gradient method",
                "recommended_for": "Most use cases, especially small to medium problems",
                "training_speed": "Fast",
                "sample_efficiency": "Good",
            },
            "alphazero": {
                "name": "AlphaZero",
                "description": "MCTS-based algorithm with neural network guidance",
                "recommended_for": "Complex problems requiring strategic planning",
                "training_speed": "Slower",
                "sample_efficiency": "Better for complex problems",
            },
        },
        "default": "ppo",
    }


@with_sync
async def get_available_policies() -> dict[str, Any]:
    """Get information about available policy network architectures.

    Returns:
        Dict with policy descriptions and recommendations
    """
    return {
        "status": "success",
        "policies": {
            "basic": {
                "name": "Basic Policy",
                "description": "Simple feedforward neural network",
                "recommended_for": "Small problems (< 8 qubits), faster training",
                "architecture": "MLP with 2-3 hidden layers",
            },
            "conv1d": {
                "name": "Conv1D Policy",
                "description": "1D convolutional neural network",
                "recommended_for": "Larger problems, when spatial structure matters",
                "architecture": "Conv1D layers followed by dense layers",
            },
        },
        "default": "basic",
    }
