from collections import defaultdict
from copy import deepcopy

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

from utils import interaction_graph

class Layout_Main(AnalysisPass):
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
        
        print(f"In Layout Main")
 
        layout = Layout()
        IG = interaction_graph(dag)
        INTER = self._get_logical_bits_interactivity(dag)
        inter_all = [(q, INTER[q]) for q in INTER]
        SPARS = self._get_physical_bits_sparsity()
        spars_all = [(v, SPARS[v]) for v in SPARS]

        while len(inter_all) > 0:
            best_log, inter = sorted(inter_all, key=lambda x: x[1], reverse=True)[0]
            best_phy, spars = sorted(spars_all, key=lambda x: x[1])[0]
            layout.add(dag.qubits[best_log], best_phy)
            inter_all.remove((best_log, inter))
            spars_all.remove((best_phy, spars))

            log_neighbors = IG[best_log]
            inter_nbr = [(n, INTER[n]) for n in log_neighbors]
            phy_neighbors = sorted(self.coupling_map.neighbors(best_phy))
            spars_nbr = [(n, SPARS[n]) for n in phy_neighbors]

            for ln, i in sorted(inter_nbr, key=lambda x: x[1], reverse=True):
                if dag.qubits[ln] not in layout.get_virtual_bits():

                    for pn, s in sorted(spars_nbr, key=lambda x: x[1]):
                        if pn not in layout.get_physical_bits():
                            layout.add(dag.qubits[ln], pn)
                            inter_all.remove((ln, i))
                            spars_all.remove((pn, s))
                            break

        self.property_set['layout'] = layout

    def _get_logical_bits_interactivity(self, dag: DAGCircuit) -> dict[int, int]:
        """Interactivity of logical qubit depends on how many times it was used in 
        the quantum circuit.
        
        Args:
            dag (DAGCircuit): DAG of quantum circuit.
        
        Returns:
            dict[int, int]: a dict of logical qubits and their interactivities.
        """
        inter = defaultdict(int)

        for q in dag.qubits:
            inter[q._index] = 0

        for gate in dag.op_nodes():
            if gate.op.num_qubits == 2:
                q0_idx, q1_idx = (dag.qubits.index(q) for q in gate.qargs)
                inter[q0_idx] += 1
                inter[q1_idx] += 1

        return inter
    
    def _get_physical_bits_sparsity(self) -> dict[int, int]:
        """Sparsity of physical qubit is defined as the longest distance between itself 
        and all the other qubits.

        Returns:
            dict[int, int]: a dict of physical qubits and their sparsities.
        """
        spars = defaultdict(int)

        for v in self.coupling_map.physical_qubits:
            spars[v] = 0

        for v0 in self.coupling_map.physical_qubits:
            for v1 in self.coupling_map.physical_qubits:
                spars[v0] = max(spars[v0], int(self.dist_matrix[v0][v1]))
        
        return spars