from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Literal, Union, Callable, Tuple

import perceval as pcvl
import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import TypeAlias  # for Python <3.10

from pcvl2torch import pcvl_circuit_to_pytorch_unitary
from slos_torch import pytorch_slos_output_distribution

class OutputMappingStrategy(Enum):
    LINEAR = 'linear'
    GROUPING = 'grouping'
    NONE = 'none'

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
        circuit (pcvl.Circuit): Perceval quantum circuit to be used
        input_state (List[int]): Initial photonic state configuration
        trainable_parameters (Union[int, List[str]], optional): Either number of trainable parameters
            or list of parameter names to make trainable. Parameters are initialized in [0, π].
        output_map_func (Callable, optional): Function to map output states
        output_mapping_strategy (OutputMappingStrategy): Strategy for mapping quantum output

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

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 circuit: pcvl.Circuit,
                 input_state: List[int],
                 trainable_parameters: Union[int, List[str]] = None,
                 output_map_func: Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]] = None,
                 output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.NONE):
        super().__init__()

        # Store circuit
        self.circuit = circuit

        self.output_map_func = output_map_func

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
                self.theta_names = [self.circuit_parameter_names[i] for i in range(trainable_parameters)]
            
            self.thetas = nn.Parameter(torch.rand(self.n_thetas) * torch.pi)
            self.circuit_parameter_map = {name: self.thetas[idx] for idx, name in enumerate(self.theta_names)}
            self.x_names = [name for name in self.circuit_parameter_names if name not in self.theta_names]
        else:
            self.circuit_parameter_map = {}
            self.thetas = None
            self.n_thetas = 0
            self.x_names = self.circuit_parameter_names

        self.n_xs = len(self.x_names)

        if len(self.x_names) != input_size:
            raise ValueError(
                f"Number of circuit inputs ({len(self.x_names)}) "
                f"must match input_size ({input_size})"
            )

        # Initialize output mapping based on strategy on a dummy distribution to discover size of output layer
        unitary = pcvl_circuit_to_pytorch_unitary(
            self.circuit, 
            torch.tensor([0.0] * self.n_circuit_parameters)
        )
        _, distribution = pytorch_slos_output_distribution(unitary, self.input_state, self.output_map_func)
        # print(distribution)
        self.setup_output_mapping(distribution)

    def setup_output_mapping(self, output_distribution):
        """Initialize output mapping based on selected strategy"""
        self.probability_distribution_size = output_distribution.shape[-1]

        if self.output_mapping_strategy == OutputMappingStrategy.LINEAR:
            self.output_mapping = nn.Linear(self.probability_distribution_size, self.output_size)
        elif self.output_mapping_strategy == OutputMappingStrategy.GROUPING:
            self.group_size = self.probability_distribution_size // self.output_size
            self.output_mapping = self.group_probabilities
        elif self.output_mapping_strategy == OutputMappingStrategy.NONE:
            if self.probability_distribution_size != self.output_size:
                raise ValueError(
                    f"Distribution size ({self.probability_distribution_size}) must equal "
                    f"output size ({self.output_size}) when using 'none' strategy"
                )
            self.output_mapping = nn.Identity()
        else:
            raise ValueError(f"Unknown output mapping strategy: {self.output_mapping_strategy}")

    def group_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Group probability distribution into equal-sized buckets"""
        pad_size = (self.output_size - (self.probability_distribution_size % self.output_size)) % self.output_size
        
        if pad_size > 0:
            padded = F.pad(probabilities, (0, pad_size))
        else:
            padded = probabilities

        if probabilities.dim() == 2:
            return padded.view(probabilities.shape[0], self.output_size, -1).sum(dim=-1)
        else:
            return padded.view(self.output_size, -1).sum(dim=-1)

    def get_quantum_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process inputs through the quantum circuit.

        Args:
            x (torch.Tensor): Input tensor [input_size] or [batch_size, input_size]

        Returns:
            torch.Tensor: Probability distribution [n_states] or [batch_size, n_states]
        """

        if x.dim() == 1:
            for idx, name in enumerate(self.x_names):
                # Scale input parameters by 2π for full phase shifter range
                self.circuit_parameter_map[name] = x[idx] * 2 * torch.pi
        else:
            for idx, name in enumerate(self.x_names):
                # Scale input parameters by 2π for full phase shifter range
                self.circuit_parameter_map[name] = x[:, idx] * 2 * torch.pi

        unitaries = pcvl_circuit_to_pytorch_unitary(self.circuit, self.circuit_parameter_map)
        _, distribution = pytorch_slos_output_distribution(unitaries, self.input_state, self.output_map_func)
        return distribution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum layer.

        Args:
            x (torch.Tensor): Input tensor [input_size] or [batch_size, input_size]

        Returns:
            torch.Tensor: Output tensor [output_size] or [batch_size, output_size]
        """
        quantum_output = self.get_quantum_output(x)
        return self.output_mapping(quantum_output)

    def __str__(self) -> str:
        """Returns a string representation of the quantum layer architecture."""
        sections = []
        
        sections.append("Quantum Neural Network Layer:")
        sections.append(f"  Input Size: {self.input_size}")
        sections.append(f"  Output Size: {self.output_size}")
        
        sections.append("Quantum Circuit Configuration:")
        sections.append(f"  Circuit: {self.circuit.describe()}")
        sections.append(f"  Number of Modes: {self.circuit.m}")
        sections.append(f"  Number of Trainable Parameters (theta): {self.n_thetas} - {', '.join(self.theta_names)}")
        sections.append(f"  Number of Inputs (x) Parameters: {self.input_size} - {', '.join(self.x_names)}")
        sections.append(f"  Input State: {self.input_state}")
        
        sections.append("\nOutput Configuration:")
        sections.append(f"  Distribution Size: {self.probability_distribution_size}")
        sections.append(f"  Output Mapping: {self.output_mapping_strategy.value}")
        if self.output_mapping_strategy == OutputMappingStrategy.GROUPING:
            sections.append(f"  Group Size: {self.group_size}")
            
        return "\n".join(sections)

# Example usage:
if __name__ == "__main__":
    # Create a simple circuit
    c = pcvl.Circuit(4)
    c.add(0, pcvl.BS()//pcvl.PS(pcvl.P("theta1"))//pcvl.BS(), merge=True)
    c.add(2, pcvl.BS()//pcvl.PS(pcvl.P("theta2"))//pcvl.BS()//pcvl.PS(pcvl.P("theta3")), merge=True)
    c.add(1, pcvl.BS()//pcvl.PS(pcvl.P("x1"))//pcvl.BS(), merge=True)

    # Create quantum layer
    qlayer = QuantumLayer(
        input_size=1,  # One input (x1)
        output_size=10,
        circuit=c,
        trainable_parameters=["theta1", "theta2", "theta3"],
        input_state=[1, 0, 1, 0],
        output_mapping_strategy=OutputMappingStrategy.LINEAR
    )

    print("--- Model description:")
    print(qlayer)

    print("--- Model parameters:")
    for name, param in qlayer.named_parameters():
        print(f"Parameter {name}: {param.shape}")

    # Test forward pass
    print("\n--- Forward Pass Example:")
    x = torch.tensor([1.0])
    print("qlayer(x)=", qlayer(x))

    # Test forward and backward pass
    x = torch.tensor([[1.0], [2.0], [3.0]])  # Batch of 3 inputs
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # Target outputs

    # Initialize optimizer
    optimizer = torch.optim.Adam(qlayer.parameters(), lr=0.01)

    # Training loop
    print("\n--- Training Example:")
    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass
        output = qlayer(x)

        # Compute loss
        loss = F.mse_loss(output, y)

        # Backward pass
        loss.backward()

        # Print gradients
        print(f"\nEpoch {epoch + 1}")
        print(f"Loss: {loss.item():.4f}")
        print("Gradients:")
        for name, param in qlayer.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm():.4f}")

        # Update parameters
        optimizer.step()


    def test_no_trainable_parameters(self, basic_circuit):
        """Test layer with no trainable parameters (pure quantum transformation)"""
        # Initialize layer with no trainable parameters
        layer = QuantumLayer(
            input_size=2,
            circuit=basic_circuit,
            input_state=[1, 0, 1, 0],
            trainable_parameters=[],  # No trainable parameters
            output_size=10
        )

        # Check that all circuit parameters are inputs
        assert len(layer.theta_names) == 0
        assert len(layer.x_names) == 4  # All circuit parameters are inputs
        assert len(list(layer.parameters())) == 0  # No trainable parameters

        # Test forward pass
        x = torch.randn(4)  # Need 4 inputs now as all parameters are inputs
        output = layer(x)
        assert output.shape == (4,)
        assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-6)

        # Test batch forward pass
        x_batch = torch.randn(3, 4)
        output_batch = layer(x_batch)
        assert output_batch.shape == (3, 4)
        assert torch.allclose(output_batch.sum(dim=1), torch.ones(3), atol=1e-6)

        # Verify no gradients are computed
        output_batch.sum().backward()
        assert all(p.grad is None for p in layer.parameters())
