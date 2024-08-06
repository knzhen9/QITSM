from collections import defaultdict
from copy import copy, deepcopy
from numpy import inf
from queue import Queue

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

from utils import (
    interaction_graph,
    dag_qubit_distance_matrix
)

class Layout_0871(AnalysisPass):
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
        """Run the SabreLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More logical qubits exist than physical.")
        
        print(f"In Layout 0871")

        best_log = self._sort_dag_qubits(dag)[0]
        best_phy = self._sort_coupling_qubits()[0]
        layout = Layout({best_log: best_phy})
        print(f"layout: {layout}")
        IG = interaction_graph(dag)
        RANK = self._get_logical_bits_rank(dag)
        log_dist_matrix = dag_qubit_distance_matrix(dag)

        # For each dag qubit, sort its neighbors based on their ranks
        for q in dag.qubits:
            ranks_nbr = [(n, RANK[n]) for n in IG[q]]
            IG[q] = [idx for idx, _ in sorted(ranks_nbr, key=lambda x: x[1])]

        p_que = Queue()
        # visited_p = dict.fromkeys(self.coupling_map.physical_qubits, False)
        # visited_v = dict.fromkeys(dag.qubits, False)

        p_que.put(best_phy)
        # visited_p[best_phy] = True
        # visited_v[best_log] = True

        while not p_que.empty():
            curr_p = p_que.get()

            # Find unmapped neighbor of current physical qubit
            for pn in sorted(self.coupling_map.neighbors(curr_p)):
                # if not visited_p[pn]:
                if pn not in layout.get_physical_bits():
                    curr_v = layout._p2v[curr_p]

                    # Find unmapped interactivity of current logical qubit
                    # If found, map it to the current p_neighbor
                    for ln in IG[curr_v]:
                        # if not visited_v[ln]:
                        if dag.qubits[ln] not in layout.get_virtual_bits():
                            # visited_v[ln] = True
                            # visited_p[pn] = True
                            layout.add(dag.qubits[ln], pn)
                            p_que.put(pn)
                            break
        
        print(f"after bfs: {layout}")

        # Sort unmapped logical and physical qubits by their distance to `best_v` and `best_p` respectively
        unmapped_log = (dag.qubits.index(q) for q in dag.qubits if q not in layout.get_virtual_bits())
        unmapped_phy = (v for v in self.coupling_map.physical_qubits if v not in layout.get_physical_bits())
        dist_to_best_log = (log_dist_matrix[q][dag.qubits.index(best_log)] for q in unmapped_log)
        sorted_unmapped_v_idx = (v_idx for _, v_idx in sorted(zip(dist_to_best_log, unmapped_log)))
        dist_to_best_phy = (self.dist_matrix[v][best_phy] for v in unmapped_phy)
        sorted_unmapped_p = (p for _, p in sorted(zip(dist_to_best_phy, unmapped_phy)))

        for v_idx, p in zip(sorted_unmapped_v_idx, sorted_unmapped_p):
            layout.add(dag.qubits[v_idx], p)

        self.property_set['layout'] = layout

    def _successors(self, node, dag):
        for _, successor, edge_data in dag.edges(node):
            if not isinstance(successor, DAGOpNode):
                continue
            if isinstance(edge_data, Qubit):
                yield successor
    
    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.applied_predecessors[node] == len(node.qargs)
    
    def _get_logical_bits_rank(self, dag: DAGCircuit):
        rank = 1
        ranks = dict.fromkeys(dag.qubits, 1)
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)

        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1

        while front_layer:
            for gate in front_layer:
                v0, v1 = gate.qargs

                if ranks[v0] == 1:
                    ranks[v0] = rank
                if ranks[v1] == 1:
                    ranks[v1] = rank
                
                front_layer.remove(gate)
                for successor in self._successors(gate, dag):
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        front_layer.append(successor)
            
            rank += 1
        
        return ranks
    
    def _sort_dag_qubits(self, dag: DAGCircuit) -> list[Qubit]:
        """Sort logical qubits of `dag` according to each qubit's interactivity.
        The first element is a logical qubit with the most interactivity.
        
        Args:
            dag (DAGCircuit): DAG to provide logical qubits and gate infos.
        """
        interact = dict.fromkeys(dag.qubits, 0)

        for gate in dag.op_nodes():
            if gate.op.num_qubits == 2:
                v0, v1 = gate.qargs
                interact[v0] += 1
                interact[v1] += 1

        return [q for q, _ in sorted(interact.items(), key=lambda x: x[1], reverse=True)]
    
    def _sort_coupling_qubits(self) -> list[int]:
        """Sort physical qubits of coupling map according to each qubit's maximum
        value of shortest distances between itself and all the other qubits.
        The first element is a physical qubit with the smallest maximum value.
        """
        max_shortest_dist = dict.fromkeys(self.coupling_map.physical_qubits, 0)

        for p0 in self.coupling_map.physical_qubits:
            for p1 in self.coupling_map.physical_qubits:
                max_shortest_dist[p0] = max(max_shortest_dist[p0], int(self.dist_matrix[p0][p1]))

        return [q for q, _ in sorted(max_shortest_dist.items(), key=lambda x: x[1])]