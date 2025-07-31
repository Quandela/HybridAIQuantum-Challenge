import torch
import json
import os

from torch.nn import Sequential

from qml.models.layers.utils import build_topology_from_list
from qml.models.topologies.register import factory
from qml.utils import make_tuple

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Dict, Tuple, Optional, Union


@dataclass_json
@dataclass
class TopologyParams:
    name: str
    input_shape: Union[Tuple, int] = field(default=None)
    output_shape: Union[Tuple, int] = field(default=None)
    extra: Optional[Dict] = field(default=None)


@dataclass
class Topology:
    __params_filename = "topology_params.json"
    __seq_list_filename = "topology_list.json"
    __weights_filename = "topology_weights.pt"

    sequence_list: List
    sequence_modules: Sequential
    params: TopologyParams = field(default=None)

    def save(self, path: str):
        with open(os.path.join(path, Topology.__seq_list_filename), "w") as out:
            seq_obj = json.dumps(self.sequence_list, indent=4)
            out.write(seq_obj)

        with open(os.path.join(path, Topology.__params_filename), "w") as out:
            out.write(self.params.to_json())

        torch.save(
            self.sequence_modules.state_dict(),
            os.path.join(path, Topology.__weights_filename),
        )

    @staticmethod
    def load(path: str) -> "Topology":
        if not os.path.isdir(path):
            print(path)
            raise Exception(path, "must be a directory")

        seq_filepath = os.path.join(path, Topology.__seq_list_filename)
        params_filepath = os.path.join(path, Topology.__params_filename)
        weight_filepath = os.path.join(path, Topology.__weights_filename)

        if not os.path.exists(seq_filepath):
            raise Exception(seq_filepath, "doesn't exist")

        if not os.path.exists(params_filepath):
            raise Exception(params_filepath, "doesn't exist")

        if not os.path.exists(weight_filepath):
            raise Exception(weight_filepath, "doesn't exist")

        with open(seq_filepath, "r") as f:
            loaded_topology_list = json.load(f)

        with open(params_filepath, "r") as f:
            loaded_params = TopologyParams.from_json(f.read())

        loaded_weights = torch.load(weight_filepath, weights_only=True)

        topology = build_topology(loaded_params, loaded_topology_list)
        topology.sequence_modules.load_state_dict(loaded_weights)

        return topology

    def get_weights(self, key: str):
        sd = self.sequence_modules.state_dict()
        if key not in sd.keys():
            raise ValueError(
                f"key {key} is not in topology. available weights: {sd.keys()}"
            )
        return sd[key]


def build_topology(
    topology_params: TopologyParams, loaded_topology_list: Optional[List] = None
) -> Topology:

    if loaded_topology_list:
        topology_pattern_list = loaded_topology_list
    else:
        topology_pattern_fct = factory()[topology_params.name]
        topology_pattern_list = topology_pattern_fct(
            input_shape=topology_params.input_shape,
            output_shape=topology_params.output_shape,
            **topology_params.extra if topology_params.extra else {},
        )

    topology_sequence, built_output_shape = build_topology_from_list(
        topology_pattern_list, topology_params.input_shape
    )

    assert (
        make_tuple(topology_params.output_shape) == built_output_shape
    ), f"Different output size. Expected:{topology_params.output_shape}, Built:{built_output_shape}"

    return Topology(
        params=topology_params,
        sequence_modules=topology_sequence,
        sequence_list=topology_pattern_list,
    )
