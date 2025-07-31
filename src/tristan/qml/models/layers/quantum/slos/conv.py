"""
This module provides functionality to convert Perceval quantum circuits to PyTorch tensors
for differentiable quantum computing.

Author: Jean Senellart

The symbolic function mapping logic is inspired by SympyTorch: https://github.com/patrick-kidger/sympytorch
    Copyright 2021 Patrick Kidger
    Licensed under the Apache License, Version 2.0 (function mapping section)
"""

import functools as ft
import numbers
from typing import Any, Callable, TypeVar, Union, Dict

import perceval as pcvl
import sympy as sp
import torch
import torch.fx as fx

# Type variable for generic function typing
T = TypeVar("T")


# Helper function to reduce multiple arguments using a binary function
def _reduce(fn: Callable[..., T]) -> Callable[..., T]:
    """
    Creates a reduction function that applies a binary operation repeatedly.
    Useful for converting n-ary Sympy operations to binary PyTorch operations.
    """

    def fn_(*args: Any) -> T:
        return ft.reduce(fn, args)

    return fn_


# Helper function to create imaginary unit tensor
def _imaginary_fnc(*_: Any) -> torch.Tensor:
    """Returns the imaginary unit as a PyTorch tensor"""
    return torch.tensor(1j)


# Mapping between Sympy operations and their PyTorch equivalents
SYMPY_TO_TORCH_OPS = {
    # Basic arithmetic
    sp.Mul: _reduce(torch.mul),
    sp.Add: _reduce(torch.add),
    sp.div: torch.div,
    sp.Pow: torch.pow,
    # Basic mathematical functions
    sp.Abs: torch.abs,
    sp.sign: torch.sign,
    sp.ceiling: torch.ceil,
    sp.floor: torch.floor,
    sp.log: torch.log,
    sp.exp: torch.exp,
    sp.sqrt: torch.sqrt,
    # Trigonometric functions
    sp.cos: torch.cos,
    sp.sin: torch.sin,
    sp.tan: torch.tan,
    sp.acos: torch.acos,
    sp.asin: torch.asin,
    sp.atan: torch.atan,
    sp.atan2: torch.atan2,
    # Hyperbolic functions
    sp.cosh: torch.cosh,
    sp.sinh: torch.sinh,
    sp.tanh: torch.tanh,
    sp.acosh: torch.acosh,
    sp.asinh: torch.asinh,
    sp.atanh: torch.atanh,
    # Complex operations
    sp.re: torch.real,
    sp.im: torch.imag,
    sp.arg: torch.angle,
    sp.core.numbers.ImaginaryUnit: _imaginary_fnc,
    sp.conjugate: torch.conj,
    # Special functions
    sp.erf: torch.erf,
    sp.loggamma: torch.lgamma,
    # Comparison operations
    sp.Eq: torch.eq,
    sp.Ne: torch.ne,
    sp.StrictGreaterThan: torch.gt,
    sp.StrictLessThan: torch.lt,
    sp.LessThan: torch.le,
    sp.GreaterThan: torch.ge,
    # Logical operations
    sp.And: torch.logical_and,
    sp.Or: torch.logical_or,
    sp.Not: torch.logical_not,
    # Min/Max operations
    sp.Max: torch.max,
    sp.Min: torch.min,
    # Matrix operations
    sp.MatAdd: torch.add,
    sp.HadamardProduct: torch.mul,
    sp.Trace: torch.trace,
    sp.Determinant: torch.det,
}


def sympy2torch(sympy_object, map_params, batch_size, dtype=torch.complex64):
    """
    Converts recursively a Sympy expression to a PyTorch tensor, expect a batch of parameters mapped in map_params.

    Args:
        sympy_object: A Sympy expression, matrix, or number
        map_params: Dictionary mapping parameter names to their PyTorch values
        batch_size: Number of samples in the batch
        dtype: PyTorch complex data type (torch.complex64 or torch.complex128)

    Returns:
        torch.Tensor: The PyTorch equivalent of the input
    """
    # Check that dtype is a complex dtype
    if dtype not in [torch.complex64, torch.complex128]:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Must be torch.complex64 or torch.complex128"
        )

    # Determine the corresponding float dtype for the complex dtype
    if dtype == torch.complex64:
        float_dtype = torch.float32
    else:  # dtype == torch.complex128
        float_dtype = torch.float64

    # Handle Perceval's matrix type
    if isinstance(sympy_object, pcvl.utils.matrix.Matrix):
        t_object = torch.empty((batch_size, *sympy_object.shape), dtype=dtype)
        for i in range(sympy_object.shape[0]):
            for j in range(sympy_object.shape[1]):
                t_object[:, i, j] = sympy2torch(
                    sympy_object[i, j], map_params, batch_size, dtype
                )

    # Handle symbolic parameters: return the corresponding tensor from map_params
    elif isinstance(sympy_object, sp.Symbol):
        t_object = map_params[sympy_object.name]
        # Ensure the tensor is of the correct dtype
        if t_object.is_complex():
            t_object = t_object.to(dtype)
        else:
            t_object = t_object.to(float_dtype)

    # Handle numerical values
    elif isinstance(sympy_object, sp.Number) or isinstance(
        sympy_object, numbers.Number
    ):
        if (
            isinstance(sympy_object, sp.Number) and sympy_object.is_real
        ) or not isinstance(sympy_object, complex):
            t_object = torch.full((batch_size,), float(sympy_object), dtype=float_dtype)
        else:
            t_object = torch.full((batch_size,), complex(sympy_object), dtype=dtype)

    # Handle operations (functions, operators) with a recursive call on the arguments
    else:
        t_object = SYMPY_TO_TORCH_OPS[sympy_object.func](
            *[
                sympy2torch(arg, map_params, batch_size=1, dtype=dtype)
                for arg in sympy_object.args
            ]
        )
        if t_object.dim() == 0:
            t_object = t_object.unsqueeze(0).repeat(batch_size)

        # Ensure correct dtype
        if t_object.is_complex():
            t_object = t_object.to(dtype)
        elif t_object.is_floating_point():
            t_object = t_object.to(float_dtype)

    return t_object


def efficient_multiply(U, cU, start_idx, end_idx):
    """
    Efficiently multiplies U_node with a matrix that would be identity with cU_node embedded at position r.

    Args:
        U: Tensor of shape (batch_size, n, n)
        cU: Tensor of shape (batch_size, k, k) where k = end_idx - start_idx
        start_idx: Starting position where cU would be embedded
        end_idx: Ending position where cU would be embedded

    Returns:
        Result of multiplication without actually creating the full embedded matrix
    """
    # Ensure both tensors are on the same device and dtype
    if U.device != cU.device:
        cU = cU.to(U.device)
    if U.dtype != cU.dtype:
        cU = cU.to(dtype=U.dtype)

    result = U.clone()  # Start with U (as if multiplying with identity)
    result[:, start_idx:end_idx, :] = torch.matmul(cU, U[:, start_idx:end_idx, :])

    return result


def build_circuit_to_unitary_fx(
    circuit: pcvl.Circuit,
    circuit_parameters: Union[torch.Tensor, Dict[str, torch.Tensor]] = {},
    device=None,
    dtype=torch.complex64,
):
    """
    Builds a torch.fx GraphModule that converts a Perceval circuit to a PyTorch unitary matrix.

    Args:
        circuit: The Perceval Circuit object to analyze
        circuit_parameters: proxy parameters for the circuit to decide on the conversion strategy
        device: PyTorch device to use for tensors. If None, will use the device of input parameters
        dtype: PyTorch complex data type (torch.complex64 or torch.complex128)

    Returns:
        fx.GraphModule: A module that converts circuit parameters to a unitary matrix

    Raises:
        ValueError: If dtype is not torch.complex64 or torch.complex128
    """
    # Validate dtype
    if dtype not in [torch.complex64, torch.complex128]:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Must be torch.complex64 or torch.complex128"
        )
    # Create a new graph
    graph = fx.Graph()

    # Add placeholder for circuit parameters, device, and dtype
    parameters_node = graph.placeholder("circuit_parameters")
    device_node = graph.placeholder("device", default_value=device)
    dtype_node = graph.placeholder("dtype", default_value=dtype)

    # Get all parameter names from the circuit to prepare dictionary access later
    circuit_param_objects = circuit.get_parameters()
    param_names = [p.name for p in circuit_param_objects]

    # Build input processing logic
    # Check if input is tensor or dict
    is_tensor_node = graph.call_function(
        lambda x: isinstance(x, torch.Tensor), (parameters_node,)
    )

    # Create a node with placeholder branches that will be selected based on input type
    def process_parameters(params, is_tensor, param_names, device, dtype):
        # Determine corresponding float dtype for the complex dtype
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

        # Determine device from parameters if not specified
        actual_device = device
        if actual_device is None:
            if is_tensor:
                actual_device = params.device
            else:
                # Get device from first tensor in dict
                for tensor in params.values():
                    if isinstance(tensor, torch.Tensor):
                        actual_device = tensor.device
                        break

        if is_tensor:
            # Handle tensor input
            if params.dim() > 2:
                raise AttributeError("torch_parameters must be a 1D or 2D tensor")

            if params.dim() == 1:
                params = params.unsqueeze(0)
                is_batch = False
            else:
                is_batch = True

            batch_size = params.size(0)

            if params.shape[-1] != len(param_names):
                raise ValueError(
                    f"Circuit requires {len(param_names)} parameters, but torch_parameters has {params.shape[-1]}"
                )

            # Create parameter mapping - ensure all on correct device and dtype
            map_params = {
                param_names[idx]: params[:, idx].to(
                    device=actual_device, dtype=float_dtype
                )
                for idx in range(len(param_names))
            }

            return map_params, batch_size, is_batch, actual_device, dtype, float_dtype
        else:
            # Handle dict input
            is_batch = False
            batch_size = 1

            # Check if any parameter tensor has batch dimension
            for name, tensor in params.items():
                if tensor.dim() == 1 or tensor.dim() == 3:
                    is_batch = True
                    batch_size = tensor.shape[0]
                    break

            # Ensure all tensors are on the correct device and dtype
            device_params = {}
            for name, tensor in params.items():
                if tensor.is_complex():
                    device_params[name] = tensor.to(device=actual_device, dtype=dtype)
                else:
                    device_params[name] = tensor.to(
                        device=actual_device, dtype=float_dtype
                    )

            return (
                device_params,
                batch_size,
                is_batch,
                actual_device,
                dtype,
                float_dtype,
            )

    # Process parameters into a standardized format
    process_node = graph.call_function(
        process_parameters,
        (parameters_node, is_tensor_node, param_names, device_node, dtype_node),
    )

    # Extract the results
    map_params_node = graph.call_function(lambda x: x[0], (process_node,))
    batch_size_node = graph.call_function(lambda x: x[1], (process_node,))
    is_batch_node = graph.call_function(lambda x: x[2], (process_node,))
    actual_device_node = graph.call_function(lambda x: x[3], (process_node,))
    actual_dtype_node = graph.call_function(lambda x: x[4], (process_node,))
    float_dtype_node = graph.call_function(lambda x: x[5], (process_node,))

    # Function to build the sympy2torch subgraph - update with device and dtype handling
    def build_sympy2torch_node(sympy_obj):
        # Handle Perceval's matrix type
        if isinstance(sympy_obj, pcvl.utils.matrix.Matrix):
            # For matrices, we'll build each element individually and then construct the matrix
            rows = []
            for i in range(sympy_obj.shape[0]):
                row_elements = []
                for j in range(sympy_obj.shape[1]):
                    element_node = build_sympy2torch_node(sympy_obj[i, j])
                    # Make sure element has correct shape [batch_size]
                    element_node = graph.call_function(
                        lambda x: x.reshape(-1), (element_node,)
                    )
                    row_elements.append(element_node)

                # Stack elements horizontally to form a row
                if len(row_elements) > 1:
                    row_node = graph.call_function(
                        torch.stack, (row_elements,), {"dim": 1}
                    )
                else:
                    row_node = graph.call_function(
                        lambda x: x.unsqueeze(1), (row_elements[0],)
                    )
                rows.append(row_node)

            # Stack rows vertically to form the matrix
            if len(rows) > 1:
                matrix_node = graph.call_function(torch.stack, (rows,), {"dim": 1})
            else:
                matrix_node = graph.call_function(lambda x: x.unsqueeze(1), (rows[0],))

            return matrix_node

        # Handle symbolic parameters
        elif isinstance(sympy_obj, sp.Symbol):
            return graph.call_function(
                lambda map_params, dtype: (
                    map_params[sympy_obj.name].unsqueeze(0)
                    if len(map_params[sympy_obj.name].shape) == 1
                    else map_params[sympy_obj.name]
                ),
                (map_params_node, actual_dtype_node),
            )

        # Handle numerical values - now with device and dtype support
        elif isinstance(sympy_obj, sp.Number) or isinstance(sympy_obj, numbers.Number):
            value = complex(sympy_obj)

            return graph.call_function(
                torch.full,
                (graph.call_function(torch.Size, ((batch_size_node,),)), value),
                {"dtype": actual_dtype_node, "device": actual_device_node},
            )

        # Handle operations (functions, operators)
        else:
            # Recursively build nodes for arguments
            arg_nodes = [build_sympy2torch_node(arg) for arg in sympy_obj.args]

            # Apply the operation
            op_func = SYMPY_TO_TORCH_OPS[sympy_obj.func]
            result_node = graph.call_function(op_func, tuple(arg_nodes))

            # Handle scalar expansion to batch
            result_node = graph.call_function(
                lambda x, bs: x.unsqueeze(0).repeat(bs) if x.dim() == 0 else x,
                (result_node, batch_size_node),
            )

            # Ensure correct dtype
            result_node = graph.call_function(
                lambda x, dtype: x.to(dtype=dtype) if x.is_complex() else x,
                (result_node, actual_dtype_node),
            )

            return result_node

    def recursive_circuit_compilation(rec_circuit):
        # Start with identity matrix - now with device and dtype handling
        u_node = graph.call_function(
            lambda batch_size, device, dtype: torch.eye(
                rec_circuit.m, dtype=dtype, device=device
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
            (batch_size_node, actual_device_node, actual_dtype_node),
        )

        # Process each component in the circuit
        for idx, (r, c) in enumerate(rec_circuit._components):
            if isinstance(circuit_parameters, dict) and c.name in circuit_parameters:
                # if parameters include a sub-circuit name, then we use the corresponding tensor
                cU_torch_node = graph.call_function(
                    lambda map_params, dtype: (
                        map_params[c.name].unsqueeze(0)
                        if len(map_params[c.name].shape) == 2
                        else map_params[c.name]
                    ),
                    (map_params_node, actual_dtype_node),
                )
            elif hasattr(c, "_components"):
                # If the component is a circuit, we need to recursively compile it
                cU_torch_node = recursive_circuit_compilation(c)
            else:
                # Get component's unitary in symbolic form and convert to computation graph
                sympy_unitary = c.compute_unitary(use_symbolic=True)
                cU_torch_node = build_sympy2torch_node(sympy_unitary)

            u_node = graph.call_function(
                efficient_multiply, (u_node, cU_torch_node, r[0], r[-1] + 1)
            )
        return u_node

    u_node = recursive_circuit_compilation(circuit)

    # Handle non-batch case
    result_node = graph.call_function(
        lambda u, is_batch: u.squeeze(0) if not is_batch else u, (u_node, is_batch_node)
    )

    # Register the output
    graph.output(result_node)

    # Create a module from this graph
    module = fx.GraphModule(torch.nn.Module(), graph)

    return module


def pcvl_circuit_to_pytorch_unitary(
    circuit: pcvl.Circuit,
    circuit_parameters: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device=None,
    dtype=torch.complex64,
):
    """
    Converts a parameterized Perceval circuit to a PyTorch unitary matrix.
    Supports batch processing if torch_parameters is a 2D tensor.

    Args:
        circuit: Perceval Circuit object
        circuit_parameters: PyTorch parameters for the circuit. Can be a 2D tensor for batch processing
                            or map name->tensor (again can be a 2D tensor)
        device: PyTorch device to use for tensors. If None, will use the device of input parameters
        dtype: PyTorch complex data type (torch.complex64 or torch.complex128)

    Returns:
        PyTorch tensor representing the circuit's unitary (or batch of unitaries)

    Raises:
        ValueError: If dtype is not torch.complex64 or torch.complex128
    """
    # Validate dtype
    if dtype not in [torch.complex64, torch.complex128]:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Must be torch.complex64 or torch.complex128"
        )
    module = build_circuit_to_unitary_fx(circuit, circuit_parameters, device, dtype)
    return module(circuit_parameters, device, dtype)


def pcvl_circuit_to_pytorch_unitary_legacy(
    circuit: pcvl.Circuit,
    circuit_parameters: Union[torch.Tensor, Dict[str, torch.Tensor]],
    dtype=torch.complex64,
):
    """
    Converts a parameterized Perceval circuit to a PyTorch unitary matrix.
    Supports batch processing if torch_parameters is a 2D tensor.

    Args:
        circuit: Perceval Circuit object
        circuit_parameters: either PyTorch parameters for the circuit. Can be a 2D tensor for batch processing.
                          or map name->tensor (again can be a 2D tensor)
        dtype: PyTorch complex data type (torch.complex64 or torch.complex128)

    Returns:
        tuple: (parameters, unitary_matrix)
            - parameters: PyTorch parameters of the circuit (or batch of parameters)
            - unitary_matrix: PyTorch tensor representing the circuit's unitary (or batch of unitaries)

    Raises:
        ValueError: If dtype is not torch.complex64 or torch.complex128
    """
    # Validate dtype
    if dtype not in [torch.complex64, torch.complex128]:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Must be torch.complex64 or torch.complex128"
        )

    # Determine the corresponding float dtype for the complex dtype
    float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

    if isinstance(circuit_parameters, torch.Tensor):
        torch_parameters = circuit_parameters
        if torch_parameters.dim() > 2:
            raise ValueError("torch_parameters must be a 1D or 2D tensor")

        # Get circuit parameters
        circuit_params = circuit.get_parameters()

        # Check if we have the correct number of parameters
        if torch_parameters.dim() == 0:
            if len(circuit_params) != 0:
                raise ValueError(
                    f"Circuit requires {len(circuit_params)} parameters, but torch_parameters has none"
                )
        elif torch_parameters.shape[-1] != len(circuit_params):
            raise ValueError(
                f"Circuit requires {len(circuit_params)} parameters, but torch_parameters has {torch_parameters.shape[-1]}"
            )

        # Initialize parameters if not provided
        is_batch = False
        if torch_parameters.dim() == 1:
            torch_parameters = torch_parameters.unsqueeze(0)  # Ensure it's a 2D tensor
        else:
            is_batch = True

        batch_size = torch_parameters.size(0)

        # Create parameter mapping and ensure correct dtype
        map_params = {
            p.name: torch_parameters[:, idx].to(dtype=float_dtype)
            for idx, p in enumerate(circuit_params)
        }
    elif isinstance(circuit_parameters, dict):
        is_batch = False
        batch_size = 1
        map_params = {}
        for name, tensor in circuit_parameters.items():
            # Convert to the appropriate dtype
            if tensor.is_complex():
                map_params[name] = tensor.to(dtype=dtype)
            else:
                map_params[name] = tensor.to(dtype=float_dtype)

            if tensor.dim() == 1:
                is_batch = True
                batch_size = tensor.shape[0]
    else:
        raise AttributeError("torch_parameters must be a map or a PyTorch tensor")

    # Build unitary matrix by composing component unitaries
    u = None
    for r, c in circuit._components:
        # TODO: we should handle recursively the case where c is a circuit, otherwise sympy unitary will be too complex

        if c.name in map_params:
            cU_torch = (
                map_params[c.name].unsqueeze(0)
                if len(map_params[c.name].shape) == 2
                else map_params[c.name]
            )
        else:
            # Get component's unitary in symbolic form
            cU = c.compute_unitary(use_symbolic=True)
            # Convert to PyTorch, returns a batch of torch unitaries
            cU_torch = sympy2torch(cU, map_params, batch_size=batch_size, dtype=dtype)

        # Handle components that don't span all modes
        if len(r) != circuit.m:
            nU = torch.eye(circuit.m, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            nU[:, r[0] : (r[-1] + 1), r[0] : (r[-1] + 1)] = cU_torch
            cU_torch = nU

        # Compose unitaries
        if u is None:
            u = cU_torch
        else:
            u = cU_torch @ u

    if not is_batch:
        u = u.squeeze(0)

    return u
