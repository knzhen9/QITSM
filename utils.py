from ag import *
from collections import defaultdict

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.coupling import CouplingMap

BACKENDS = {
    'tokyo': tokyo(),
    'rochester': rochester(),
    'sycamore': sycamore53()
}

def coupling_longest_shortest_distance(coupling_map: CouplingMap):
# TODO: check the type of coupling_map in the function rather than in the argument
    max_shortest_dist = dict.fromkeys(coupling_map.physical_qubits, 0)
    dist_matrix = coupling_map.distance_matrix

    for p0 in coupling_map.physical_qubits:
        for p1 in coupling_map.physical_qubits:
            max_shortest_dist[p0] = max(max_shortest_dist[p0], int(dist_matrix[p0][p1]))
    
    return max_shortest_dist

def coupling_qubit_neighborhood(coupling_map: CouplingMap, center: int, range: int) -> list[int]:
# TODO: check the type of coupling_map in the function rather than in the argument
    """Get the physical qubits which are `range` or less hops away from `center`.
    
    Args:
        center (int): the physical qubit to find its neighborhood.
        range (int): the maximum hops away from `center`.
    
    Returns:
        list[int]: The neighborhood of `center` (excluding itself) sorted by
        the distance away from `center`.
    """
    neighborhood = list()
    dist_matrix = coupling_map.distance_matrix

    for p in coupling_map.physical_qubits:
        dist = int(dist_matrix[center][p])
        if dist != 0 and dist <= range:
            neighborhood.append((dist, p))
    
    return [p for _, p in sorted(neighborhood)]

def interaction_graph(dag: DAGCircuit) -> dict[Qubit, set]:
# TODO: check the type of dag in the function rather than in the argument
    """Iterate all logical qubits while collecting whom current qubit has interactivity with.

    Args:
        dag (DAGCircuit): DAG to provide all gate info.
    
    Returns:
        dict[Qubit, set]: every key is a `dag` qubit, and the corresponding value is a set of
        logical qubits that have interactivity with the qubit in key.
    """
    ig = defaultdict(list)

    for q in dag.qubits:
        ig[q._index] = []

    for gate in dag.op_nodes():
        if gate.op.num_qubits == 2:
            q0_idx, q1_idx = (dag.qubits.index(q) for q in gate.qargs)

            if q1_idx not in ig[q0_idx]:
                ig[q0_idx].append(q1_idx)
                ig[q1_idx].append(q0_idx)
    
    return ig

def dag_qubit_distance_matrix(dag: DAGCircuit):
# TODO: check the type of dag in the function rather than in the argument

    dist_matrix = [[0] * len(dag.qubits) for _ in range(len(dag.qubits))]
    interact = interaction_graph(dag)

    for v in dag.qubits:
        queue = list()
        visited = dict.fromkeys(dag.qubits, False)
        queue.append(v)
        visited[v] = True

        while queue:
            curr_v = queue.pop(0)
            curr_v_idx = dag.qubits.index(curr_v)

            for neighbor in interact[curr_v]:
                neighbor_idx = dag.qubits.index(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    v_idx = dag.qubits.index(v)
                    dist_matrix[v_idx][neighbor_idx] = dist_matrix[v_idx][curr_v_idx] + 1
                    queue.append(neighbor)
    
    return dist_matrix