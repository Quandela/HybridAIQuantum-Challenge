import perceval as pcvl
import torch
import torch.nn.functional as F

from enum import Enum
from typing import List, Optional, Union, Callable, Tuple, Dict
from torch import nn

from .slos_torchscript import build_slos_compute_graph
from .conv import build_circuit_to_unitary_fx


class OutputMappingStrategy(Enum):
    LINEAR = "linear"
    GROUPING = "grouping"
    NONE = "none"


class QuantumLayer(nn.Module):
    """

    Quantum Neural Network Layer implemented using photonic circuits.

    The layer consists of a parameterized quantum photonic circuit where:
    - Some circuit parameters (in Perceval terminology) are trainable parameters (theta)
    - Others are inputs (x) fed during the forward pass
    - The output is a probability distribution over possible photonic states

    Parameter Ranges:
    - Input parameters (x) should be in range [0, 1]. These values are internally scaled
      by 2π when setting phase shifters to utilize their full range.
    - Trainable parameters (theta) are initialized in range [0, π] and will adapt during training
      to optimize the circuit's behavior.

    The output mapping strategy determines how the quantum probability distribution
    is mapped to the final output:
    - 'linear': Applies a trainable linear layer
    - 'grouping': Groups distribution values into equal-sized buckets
    - 'none': No mapping (requires matching sizes between probability distribution and output)

    Args:
        input_size (int): Number of input variables for the circuit
        output_size (int): Dimension of the final layer output
        circuit (pcvl.Circuit): Perceval quantum circuit to be used - this circuit can be changed dynamically but the
            size and parameters shall remain identical
        input_state (List[int]): Initial photonic state configuration - the input state can be changed dynamically but
            the number of photons shall remain identical
        trainable_parameters (Union[int, List[str]], optional): Either number of trainable parameters
            or list of parameter names to make trainable. Parameters are initialized in [0, π].
        output_map_func (Callable, optional): Function to map output states
        output_mapping_strategy (OutputMappingStrategy): Strategy for mapping quantum output
        device (torch.device, optional): Device to run computations on
        dtype (torch.dtype, optional): Numerical precision for computations

    Raises:
        ValueError: If input state size doesn't match circuit modes
        ValueError: If output_mapping_strategy is 'none' and distribution size != output_size

    Note:
        Input parameters (x) shall be normalized to [0, 1] range. The layer internally scales
        these values by 2π when applying them to phase shifters. This ensures full coverage
        of the phase shifter range while maintaining a normalized input interface.

    Example:
        >>> layer = QuantumLayer(
        ...     input_size=4,
        ...     output_size=4,
        ...     circuit=pcvl.Circuit(2)//pcvl.BS()//pcvl.PS(pcvl.P('theta1'))//pcvl.BS()//pcvl.PS(pcvl.P('x1'))//pcvl.BS(),
        ...     input_state=[1, 1, 1],
        ...     trainable_parameters=1,
        ...     output_mapping_strategy=OutputMappingStrategy.LINEAR
        ... )
        >>> x = torch.tensor([0.5])  # Input in [0, 1] range, will be scaled by 2π
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        circuit: pcvl.Circuit,
        input_state: List[int],
        trainable_parameters: Union[int, List[str]] = None,
        no_bunching: bool = False,
        output_map_func: Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]] = None,
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.NONE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Store circuit
        self.circuit = circuit
        self.device = device
        # Determine appropriate complex dtype based on input dtype
        if dtype is None:
            self.dtype = torch.complex64
        elif dtype == torch.float32:
            self.dtype = torch.complex64
        elif dtype == torch.float64:
            self.dtype = torch.complex128
        else:
            self.dtype = torch.complex64  # Default to complex64

        self.output_map_func = output_map_func
        self.no_bunching = no_bunching

        self.circuit_parameters = self.circuit.get_parameters()
        self.n_circuit_parameters = len(self.circuit_parameters)
        self.circuit_parameter_names = [p.name for p in self.circuit_parameters]

        # Validate input state
        self.input_state = input_state
        if len(self.input_state) != self.circuit.m:
            raise ValueError(
                "Input state size must match number of modes in the circuit"
            )

        # Setup trainable parameters and inputs
        self.input_size = input_size
        self.output_size = output_size
        self.output_mapping_strategy = output_mapping_strategy

        if trainable_parameters is not None:
            if isinstance(trainable_parameters, list):
                self.n_thetas = len(trainable_parameters)
                self.theta_names = trainable_parameters
            else:
                self.n_thetas = trainable_parameters
                self.theta_names = [
                    self.circuit_parameter_names[i] for i in range(trainable_parameters)
                ]

            self.thetas = nn.Parameter(torch.rand(self.n_thetas) * torch.pi)
            self.x_names = [
                name
                for name in self.circuit_parameter_names
                if name not in self.theta_names
            ]
        else:
            self.thetas = None
            self.n_thetas = 0
            self.x_names = self.circuit_parameter_names

        self.n_xs = len(self.x_names)

        if len(self.x_names) != input_size:
            raise ValueError(
                f"Number of circuit inputs ({len(self.x_names)}) "
                f"must match input_size ({input_size})"
            )

        # Initialize computation graphs - using dictionary format for parameter passing
        # Create a parameter dict for initialization
        init_params = {}
        # Add trainable parameters to initialization dict
        if self.thetas is not None:
            for idx, name in enumerate(self.theta_names):
                init_params[name] = torch.zeros(1)  # Placeholder for graph building

        # Add input parameters to initialization dict
        for name in self.x_names:
            init_params[name] = torch.zeros(1)  # Placeholder for graph building

        # Build computation graph for unitary conversion
        self.unitary_graph = build_circuit_to_unitary_fx(
            self.circuit, init_params, device=device, dtype=self.dtype
        )

        # Build computation graph for output distribution calculation
        # Convert complex dtype to corresponding real dtype
        if self.dtype == torch.complex64:
            real_dtype = torch.float32
        elif self.dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            real_dtype = torch.float32

        self.simulation_graph = build_slos_compute_graph(
            self.input_state,
            output_map_func=output_map_func,
            no_bunching=no_bunching,
            device=device,
            dtype=real_dtype,
        )

        # Initialize output mapping
        # First, create a dummy distribution to determine its size
        dummy_params = {name: torch.zeros(1) for name in self.circuit_parameter_names}
        unitary = self.unitary_graph(dummy_params)
        _, distribution = self.simulation_graph.compute(unitary)
        self.setup_output_mapping(distribution)

    def change_circuit(self, circuit: pcvl.Circuit):
        """
        Dynamically change the quantum circuit while maintaining compatibility with the layer configuration.

        The new circuit must have the same number of modes and identical parameter names as the original
        circuit to ensure compatibility with the existing layer configuration and trained parameters.

        Args:
            circuit (pcvl.Circuit): New Perceval quantum circuit to replace the current one

        Raises:
            ValueError: If the new circuit has a different number of modes
            ValueError: If the new circuit has a different number of parameters
            ValueError: If the new circuit's parameter names don't match the original circuit

        Example:
            >>> new_circuit = pcvl.Circuit(4)  # Same number of modes as original
            >>> new_circuit.add(0, pcvl.BS()//pcvl.PS(pcvl.P("theta1"))//pcvl.BS())
            >>> layer.change_circuit(new_circuit)  # Updates circuit while preserving parameters
        """
        if self.circuit.m != circuit.m:
            raise ValueError("New circuit must have the same number of modes")

        new_circuit_parameters = circuit.get_parameters()

        if len(new_circuit_parameters) != self.n_circuit_parameters:
            raise ValueError("New circuit must have the same number of parameters")

        old_circuit_params = set(p.name for p in self.circuit_parameters)
        new_circuit_parameters = set(p.name for p in new_circuit_parameters)

        if old_circuit_params != new_circuit_parameters:
            raise ValueError("New circuit parameters must match the original circuit")

        # Update circuit
        self.circuit = circuit

        # Rebuild unitary computation graph
        init_params = {}
        if self.thetas is not None:
            for idx, name in enumerate(self.theta_names):
                init_params[name] = torch.zeros(1)

        for name in self.x_names:
            init_params[name] = torch.zeros(1)

        self.unitary_graph = build_circuit_to_unitary_fx(
            self.circuit, init_params, device=self.device, dtype=self.dtype
        )

    def change_input_state(self, input_state: List[int]):
        """Change the input state while keeping the same number of modes and photons"""
        if len(input_state) != self.circuit.m:
            raise ValueError(
                "New input state must match the number of modes in the circuit"
            )

        if sum(input_state) != sum(self.input_state):
            raise ValueError("New input state must have the same number of photons")

        self.input_state = input_state

        # Rebuild simulation computation graph
        # Convert complex dtype to corresponding real dtype
        if self.dtype == torch.complex64:
            real_dtype = torch.float32
        elif self.dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            real_dtype = torch.float32

        self.simulation_graph = build_slos_compute_graph(
            self.input_state,
            output_map_func=self.output_map_func,
            no_bunching=self.no_bunching,
            device=self.device,
            dtype=real_dtype,
        )

    def setup_output_mapping(self, output_distribution):
        """Initialize output mapping based on selected strategy"""
        self.probability_distribution_size = output_distribution.shape[-1]

        if self.output_mapping_strategy == OutputMappingStrategy.LINEAR:
            self.output_mapping = nn.Linear(
                self.probability_distribution_size, self.output_size
            )
        elif self.output_mapping_strategy == OutputMappingStrategy.GROUPING:
            self.group_size = self.probability_distribution_size // self.output_size
            self.output_mapping = self.group_probabilities
        elif self.output_mapping_strategy == OutputMappingStrategy.NONE:
            self.output_size = self.probability_distribution_size
            # if self.probability_distribution_size != self.output_size:
            # raise ValueError(
            #     f"Distribution size ({self.probability_distribution_size}) must equal "
            #     f"output size ({self.output_size}) when using 'none' strategy"
            # )
            self.output_mapping = nn.Identity()
        else:
            raise ValueError(
                f"Unknown output mapping strategy: {self.output_mapping_strategy}"
            )

    def group_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Group probability distribution into equal-sized buckets"""
        pad_size = (
            self.output_size - (self.probability_distribution_size % self.output_size)
        ) % self.output_size

        if pad_size > 0:
            padded = F.pad(probabilities, (0, pad_size))
        else:
            padded = probabilities

        if probabilities.dim() == 2:
            return padded.view(probabilities.shape[0], self.output_size, -1).sum(dim=-1)
        else:
            return padded.view(self.output_size, -1).sum(dim=-1)

    def prepare_parameters(self, x: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare parameter dictionary for circuit evaluation.

        Args:
            x (torch.Tensor): Input tensor [input_size] or [batch_size, input_size]

        Returns:
            Dict[str, torch.Tensor]: Parameter dictionary for circuit evaluation
        """
        params = {}

        # Add trainable parameters to dict
        if self.thetas is not None:
            for idx, name in enumerate(self.theta_names):
                params[name] = self.thetas[idx]

        # Add input parameters (x) to dict if provided
        if x is not None:
            if x.dim() == 1:
                for idx, name in enumerate(self.x_names):
                    # Scale input parameters by 2π for full phase shifter range
                    params[name] = x[idx] * 2 * torch.pi
            else:
                # Handle batched inputs - we need to properly prepare parameter dict
                batch_size = x.shape[0]

                # First, convert trainable parameters to batched form if needed
                if self.thetas is not None:
                    for idx, name in enumerate(self.theta_names):
                        # Expand trainable parameters to match batch size
                        params[name] = self.thetas[idx].expand(batch_size)

                # Add input parameters
                for idx, name in enumerate(self.x_names):
                    params[name] = x[:, idx] * 2 * torch.pi

        return params

    def get_quantum_output(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Process inputs through the quantum circuit.

        Args:
            x (torch.Tensor): Input tensor [input_size] or [batch_size, input_size]

        Returns:
            torch.Tensor: Probability distribution [n_states] or [batch_size, n_states]
        """
        # Prepare parameter dictionary
        params = self.prepare_parameters(x)

        # Compute unitary using computation graph
        unitaries = self.unitary_graph(params)

        # Compute output distribution using simulation graph
        keys, distribution = self.simulation_graph.compute(unitaries)

        return keys, distribution

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the quantum layer.

        Args:
            x (torch.Tensor): Input tensor [input_size] or [batch_size, input_size]

        Returns:
            torch.Tensor: Output tensor [output_size] or [batch_size, output_size]
        """
        keys, quantum_output = self.get_quantum_output(x)
        return keys, self.output_mapping(quantum_output)

    def __str__(self) -> str:
        """Returns a string representation of the quantum layer architecture."""
        sections = []

        sections.append("Quantum Neural Network Layer:")
        sections.append(f"  Input Size: {self.input_size}")
        sections.append(f"  Output Size: {self.output_size}")

        sections.append("Quantum Circuit Configuration:")
        sections.append(f"  Circuit: {self.circuit.describe()}")
        sections.append(f"  Number of Modes: {self.circuit.m}")
        sections.append(
            f"  Number of Trainable Parameters (theta): {self.n_thetas} - {', '.join(self.theta_names) if self.theta_names else 'None'}"
        )
        sections.append(
            f"  Number of Inputs (x) Parameters: {self.input_size} - {', '.join(self.x_names)}"
        )
        sections.append(f"  Input State: {self.input_state}")
        sections.append(f"  Device: {self.device}")
        sections.append(f"  Data Type: {self.dtype}")

        sections.append("\nOutput Configuration:")
        sections.append(f"  Distribution Size: {self.probability_distribution_size}")
        sections.append(f"  Output Mapping: {self.output_mapping_strategy.value}")
        if self.output_mapping_strategy == OutputMappingStrategy.GROUPING:
            sections.append(f"  Group Size: {self.group_size}")

        return "\n".join(sections)
