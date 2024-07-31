from collections import defaultdict
from copy import copy, deepcopy
from numpy import inf
from queue import Queue

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

from utils import (
    coupling_longest_shortest_distance,
    coupling_qubit_neighborhood,
    interaction_graph,
    dag_qubit_distance_matrix
)

class Layout_new(AnalysisPass):
    def __init__(self, coupling_map):
        super().__init__()

        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

        if self.coupling_map is not None:
            if not self.coupling_map.is_symmetric:
                if isinstance(coupling_map, CouplingMap):
                    self.coupling_map = deepcopy(self.coupling_map)
                self.coupling_map.make_symmetric()

        self.dist_matrix = self.coupling_map.distance_matrix

    def run(self, dag: DAGCircuit):
        """Run the Layout_new pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More logical qubits exist than physical.")
        
        print(f"In Layout new")
 
        layout = Layout()
        IG = interaction_graph(dag)
        INTER = self._get_logical_bits_interactivity(dag)
        inter_all = [(q, INTER[q]) for q in INTER]
        # print(f"inter_all: {inter_all}")
        SPARS = self._get_physical_bits_sparsity()
        spars_all = [(v, SPARS[v]) for v in SPARS]
        # print(f"spars_all: {spars_all}")

        while len(inter_all) > 0:
            best_log, inter = sorted(inter_all, key=lambda x: x[1], reverse=True)[0]
            best_phy, spars = sorted(spars_all, key=lambda x: x[1])[0]
            layout.add(dag.qubits[best_log], best_phy)
            inter_all.remove((best_log, inter))
            # print(f"best log: ({best_log}, {inter})")
            # print(f"inter_all: {inter_all}")
            spars_all.remove((best_phy, spars))

            log_neighbors = IG[best_log]
            inter_nbr = [(n, INTER[n]) for n in log_neighbors]
            # print(f"inter_nbr: {inter_nbr}")
            # print(f"sorted inter_nbr: {sorted(inter_nbr, key=lambda x: x[1], reverse=True)}")
            phy_neighbors = sorted(self.coupling_map.neighbors(best_phy))
            spars_nbr = [(n, SPARS[n]) for n in phy_neighbors]
            # print(f"spars_nbr: {spars_nbr}")

            for ln, i in sorted(inter_nbr, key=lambda x: x[1], reverse=True):
                if dag.qubits[ln] not in layout.get_virtual_bits():

                    for pn, s in sorted(spars_nbr, key=lambda x: x[1]):
                        if pn not in layout.get_physical_bits():
                            layout.add(dag.qubits[ln], pn)
                            # print(f"neighbor: ({ln}, {i})")
                            inter_all.remove((ln, i))
                            # print(f"after remove neighbor: {inter_all}")
                            spars_all.remove((pn, s))
                            break

        print(f"layout: {layout}")
        self.property_set['layout'] = layout

    def _get_logical_bits_interactivity(self, dag: DAGCircuit) -> dict[int, int]:
        """Interactivity of logical qubit depends on how many times it was used in 
        the quantum circuit.
        
        Args:
            dag (DAGCircuit): DAG of quantum circuit.
        
        Returns:
            dict[int, int]: a dict of logical qubits and their interactivities.
        """
        # inter = dict.fromkeys(list(range(len(dag.qubits))), 0)
        inter = defaultdict(int)
        print(inter)

        for q in dag.qubits:
            inter[q._index] = 0

        for gate in dag.op_nodes():
            if gate.op.num_qubits == 2:
                q0_idx, q1_idx = (dag.qubits.index(q) for q in gate.qargs)
                inter[q0_idx] += 1
                inter[q1_idx] += 1

        print(f"inter: {inter}")
        return inter
        inter = sorted(inter.items(), key=lambda x: x[1], reverse=True)
        print(inter)
        # return inter
    
    def _get_physical_bits_sparsity(self) -> dict[int, int]:
        """Sparsity of physical qubit is defined as the longest distance between itself 
        and all the other qubits.

        Returns:
            dict[int, int]: a dict of physical qubits and their sparsities.
        """
        # spars = dict.fromkeys(self.coupling_map.physical_qubits, 0)
        spars = defaultdict(int)
        print(spars)

        for v in self.coupling_map.physical_qubits:
            spars[v] = 0

        for v0 in self.coupling_map.physical_qubits:
            for v1 in self.coupling_map.physical_qubits:
                spars[v0] = max(spars[v0], int(self.dist_matrix[v0][v1]))
        
        print(f"spars: {spars}")
        return spars
        spars = sorted(spars.items(), key=lambda x: x[1])
        print(spars)
        return spars