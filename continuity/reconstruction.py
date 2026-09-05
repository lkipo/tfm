"""
Vascular tree reconstruction from a 3D skeleton point cloud.

Two methods live here (see documentation/README_LABEL_CONNECTED.md and
documentation/lab_diary.md for the full reference):

  * Basic CCO          -- `VascularTreeReconstruction` / `MultiTreeReconstruction`
  * Label-connected    -- `LabelConnectedMultiTreeReconstruction` (recommended)

`ImprovedVascularTreeReconstruction` is not a standalone method: it exists
because label-connected reconstruction uses it for the per-group tree maths.

Everything is in the voxel-index space of the input point cloud; no affine or
spacing transform is applied anywhere (lab_diary.md 5.3).
"""

import json

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


class VascularTreeReconstruction:
    """
    Reconstruct vascular tree from skeleton points using CCO principles
    """

    def __init__(self, skeleton_points, gamma=3.0, mu=3.6e-3,
                 Q_perf=0.125, P_out=60, P_in=100):
        """
        Parameters:
        -----------
        skeleton_points : np.array, shape (N, 3)
            3D coordinates of skeleton points
        gamma : float
            Murray's law exponent (default: 3.0)
        mu : float
            Blood viscosity in Pa·s (default: 3.6e-3 Pa·s = 3.6 cP)
        Q_perf : float
            Flow at each terminal in mL/min (default: 0.125)
        P_out : float
            Pressure at terminals in mmHg (default: 60)
        P_in : float
            Pressure at root in mmHg (default: 100)
        """
        self.points = np.array(skeleton_points)
        self.gamma = gamma
        self.mu = mu
        self.Q_perf = (Q_perf * 1000) / 60  # Convert mL/min to mm³/s
        self.P_out = P_out * 133.322  # Convert mmHg to Pa
        self.P_in = P_in * 133.322
        self.kappa = 8 * mu / np.pi
        self.xi = self.Q_perf / (self.P_in - self.P_out)

        self.graph = None
        self.root = None
        self.tree_structure = None

    def build_graph_from_skeleton(self, k_neighbors=10, max_edge_length=None):
        """
        Step 1-2: Build connected graph from skeleton points
        Uses k-nearest neighbors + MST to ensure connectivity

        Parameters:
        -----------
        k_neighbors : int
            Number of nearest neighbors to consider
        max_edge_length : float
            Maximum allowed edge length (for pruning)
        """
        print("Building graph from skeleton points...")
        n_points = len(self.points)

        # Build k-NN graph
        tree = cKDTree(self.points)

        # Create adjacency matrix
        distances = np.zeros((n_points, n_points))
        distances[:] = np.inf

        for i in range(n_points):
            dists, indices = tree.query(
                self.points[i], k=min(k_neighbors+1, n_points))
            for j, idx in enumerate(indices[1:]):  # Skip self
                distances[i, idx] = dists[j+1]
                distances[idx, i] = dists[j+1]  # Symmetric

        # Build MST to ensure single connected component and remove cycles
        mst = minimum_spanning_tree(distances)

        # Convert to networkx graph
        self.graph = nx.Graph()

        # Add nodes with positions
        for i, pos in enumerate(self.points):
            self.graph.add_node(i, pos=pos)

        # Add edges from MST
        mst_coo = mst.tocoo()
        for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            self.graph.add_edge(i, j, length=weight)


        print(
            f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # Prune short branches
        if max_edge_length:
            self._prune_branches(max_edge_length)

        return self.graph

    def _prune_branches(self, min_length):
        """Remove short terminal branches"""
        pruned = True
        while pruned:
            pruned = False
            endpoints = [n for n in self.graph.nodes()
                         if self.graph.degree(n) == 1]

            for node in endpoints:
                neighbor = list(self.graph.neighbors(node))[0]
                edge_length = self.graph[node][neighbor]['length']

                if edge_length < min_length:
                    self.graph.remove_node(node)
                    pruned = True

        print(
            f"After pruning: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def find_root_candidates(self, n_candidates=5, method="highest_z"):
        """
        Step 4: Identify potential root points

        Returns candidates based on:
        - Boundary points (degree 1)
        - Spatial location (e.g., superior, medial)
        - High centrality
        """
        candidates = []

        # Get all endpoints (degree 1)
        endpoints = [n for n in self.graph.nodes()
                     if self.graph.degree(n) == 1]

        if len(endpoints) == 0:
            print("Warning: No endpoints found, using highest degree nodes")
            degrees = dict(self.graph.degree())
            endpoints = sorted(degrees, key=degrees.get, reverse=True)[:10]

        endpoint_positions = np.array([self.points[i] for i in endpoints])

        if method == "highest_z":
            # Strategy 1: Highest z-coordinate (superior position for hepatic vein)
            z_coords = endpoint_positions[:, 2]
            top_z_indices = np.argsort(z_coords)[-n_candidates:]
            candidates.extend([endpoints[i] for i in top_z_indices])

            print(
                f"Found {len(candidates)} candidates from superior endpoints")

        if method == "lowest_z":
            # Strategy 1: Lowest z-coordinate (inferior position for hepatic vein)
            z_coords = endpoint_positions[:, 2]
            bottom_z_indices = np.argsort(z_coords)[:n_candidates]
            candidates.extend([endpoints[i] for i in bottom_z_indices])

            print(
                f"Found {len(candidates)} candidates from inferior endpoints")

        if method == "centrality":
            # Strategy 2: Most central points (low eccentricity)
            if len(self.graph) > 10:
                try:
                    # Get largest connected component
                    if not nx.is_connected(self.graph):
                        largest_cc = max(
                            nx.connected_components(self.graph), key=len)
                        subgraph = self.graph.subgraph(largest_cc)
                    else:
                        subgraph = self.graph

                    centrality = nx.closeness_centrality(subgraph)
                    central_nodes = sorted(centrality, key=centrality.get, reverse=True)[
                        :n_candidates]
                    candidates.extend(
                        [n for n in central_nodes if n in endpoints])
                except:
                    pass

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        print(f"Found {len(unique_candidates)} root candidates")
        return unique_candidates[:n_candidates]

    def orient_tree_from_root(self, root):
        """
        Step 5a: Create directed tree structure from root
        Returns tree with parent-child relationships
        """
        if not nx.is_connected(self.graph):
            # Get largest connected component containing root
            for component in nx.connected_components(self.graph):
                if root in component:
                    subgraph = self.graph.subgraph(component).copy()
                    break
        else:
            subgraph = self.graph

        # BFS from root to create directed tree
        tree = nx.DiGraph()
        visited = {root}
        queue = [root]

        # Copy node attributes
        for node in subgraph.nodes():
            tree.add_node(node, pos=self.points[node])

        while queue:
            current = queue.pop(0)

            for neighbor in subgraph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Add directed edge from parent to child
                    edge_data = subgraph[current][neighbor]
                    tree.add_edge(current, neighbor, **edge_data)
                    queue.append(neighbor)

        return tree

    def compute_tree_parameters(self, tree, root):
        """
        Step 5b: Compute L_i (number of terminals in subtree)
        """
        # Find all terminal nodes (leaves)
        terminals = [n for n in tree.nodes() if tree.out_degree(n)
                     == 0 and n != root]

        # Compute L_i for each node (number of terminals in subtree)
        L = {}

        def count_terminals(node):
            if tree.out_degree(node) == 0:  # Terminal
                L[node] = 1
                return 1

            total = 0
            for child in tree.successors(node):
                total += count_terminals(child)
            L[node] = total
            return total

        count_terminals(root)

        return L, terminals

    def compute_radii(self, tree, root, L):
        """
        Step 5c: Apply CCO radius formulas (Equations 1-7)
        """
        # Initialize radius dict
        R = {}  # Hydraulic resistance
        beta = {}  # Radius ratio to parent
        rho = {}  # Relative radius to root segment
        radii = {}  # Actual radii

        # Compute resistance R and beta bottom-up
        def compute_resistance(node):
            children = list(tree.successors(node))

            if len(children) == 0:  # Terminal node
                # For terminal: R = kappa * length / r^4
                # We'll initialize with a default and iterate
                R[node] = 1.0  # Will be updated
                return R[node]

            # Get edge length to parent
            parent = list(tree.predecessors(node))
            if parent:
                length = tree[parent[0]][node]['length']
            else:  # Root
                # Estimate length from first child
                if children:
                    length = tree[node][children[0]]['length']
                else:
                    length = 1.0

            # Compute children resistances first (bottom-up)
            child_resistances = []
            for child in children:
                child_resistances.append(compute_resistance(child))

            # Compute beta for each child (Equations 3-4)
            if len(children) == 2:
                child1, child2 = children

                # Alpha ratio (Eq. 4)
                alpha = ((L[child1] / L[node]) *
                         (R[child1] / R[child2])) ** 0.25

                # Beta for each child (Eq. 3)
                beta[child1] = (1 + alpha ** self.gamma) ** (-1/self.gamma)
                beta[child2] = (1 + (1/alpha) ** self.gamma) ** (-1/self.gamma)
            else:
                # Single child or more than 2 children
                for child in children:
                    beta[child] = 1.0

            # Compute resistance (Eq. 6)
            sum_term = sum(beta[child]**4 / R[child] for child in children)
            if sum_term > 0:
                R[node] = self.kappa * length + (1 / sum_term)
            else:
                R[node] = self.kappa * length

            return R[node]

        # Start computation from root
        compute_resistance(root)

        # Compute absolute radii top-down (Eq. 7)
        def compute_radii_recursive(node, parent_radius=None):
            if parent_radius is None:
                # Root radius (Eq. 7)
                n_terminals = L[root]
                r1 = (self.xi * R[root] * n_terminals) ** 0.25
                radii[node] = r1
                rho[node] = 1.0
            else:
                # Child radius
                rho[node] = rho[list(tree.predecessors(node))[0]] * beta[node]
                radii[node] = parent_radius * beta[node]

            # Recurse to children
            for child in tree.successors(node):
                compute_radii_recursive(child, radii[node])

        compute_radii_recursive(root)

        return radii, R, beta, rho

    def compute_radii_iterative(self, tree, root, L, max_iterations=15, tol=1e-6,
                                 initial_terminal_radius=1.0):
        """
        Step 5c (fixed): iteratively solve for radii using the terminal-resistance
        fixed point that `compute_radii`'s own comment describes but never performs.

        `compute_radii` hardcodes terminal resistance to a placeholder `R[node] = 1.0`
        ("will be updated" -- it never is), even though physically
        R_terminal = kappa * length / r**4. Since r is exactly what we're solving for,
        this is a fixed-point problem: guess each terminal's radius, compute
        resistances/radii top-to-bottom as usual, feed the newly computed terminal
        radii back in as the next guess, and repeat until they stop changing.

        This mainly affects the branching ratio at the last bifurcation before each
        terminal: with the old flat R=1.0, every terminal looks identical to its
        sibling regardless of its actual segment length, which biases the estimated
        radii exactly where bifurcations are most numerous (near the leaves).

        Returns the same (radii, R, beta, rho) tuple as `compute_radii`, so it's a
        drop-in alternative; nothing about `compute_radii` itself is modified.
        """
        terminals = [n for n in tree.nodes()
                     if tree.out_degree(n) == 0 and n != root]

        # Seed guess for each terminal's own radius (only used to seed R_terminal).
        terminal_radius = {t: initial_terminal_radius for t in terminals}

        radii, R, beta, rho = {}, {}, {}, {}

        for _ in range(max_iterations):
            R = {}
            beta = {}
            rho = {}
            radii = {}

            def compute_resistance(node):
                children = list(tree.successors(node))

                if len(children) == 0:  # Terminal node
                    parent = list(tree.predecessors(node))
                    length = tree[parent[0]][node]['length'] if parent else 1.0
                    r_term = max(terminal_radius[node], 1e-9)
                    R[node] = self.kappa * length / r_term**4
                    return R[node]

                parent = list(tree.predecessors(node))
                if parent:
                    length = tree[parent[0]][node]['length']
                else:  # Root
                    length = tree[node][children[0]
                                         ]['length'] if children else 1.0

                for child in children:
                    compute_resistance(child)

                if len(children) == 2:
                    child1, child2 = children
                    alpha = ((L[child1] / L[node]) *
                             (R[child1] / R[child2])) ** 0.25
                    beta[child1] = (1 + alpha ** self.gamma) ** (-1/self.gamma)
                    beta[child2] = (1 + (1/alpha) **
                                     self.gamma) ** (-1/self.gamma)
                else:
                    for child in children:
                        beta[child] = 1.0

                sum_term = sum(beta[child]**4 / R[child]
                                for child in children)
                if sum_term > 0:
                    R[node] = self.kappa * length + (1 / sum_term)
                else:
                    R[node] = self.kappa * length

                return R[node]

            compute_resistance(root)

            def compute_radii_recursive(node, parent_radius=None):
                if parent_radius is None:
                    n_terminals = L[root]
                    r1 = (self.xi * R[root] * n_terminals) ** 0.25
                    radii[node] = r1
                    rho[node] = 1.0
                else:
                    rho[node] = rho[list(tree.predecessors(node))[0]] * beta[node]
                    radii[node] = parent_radius * beta[node]

                for child in tree.successors(node):
                    compute_radii_recursive(child, radii[node])

            compute_radii_recursive(root)

            if not terminals:
                break

            new_terminal_radius = {t: radii[t] for t in terminals}
            max_change = max(abs(new_terminal_radius[t] - terminal_radius[t])
                              for t in terminals)
            terminal_radius = new_terminal_radius

            if max_change < tol:
                break

        return radii, R, beta, rho

    def optimize_bifurcation(self, tree, node, radii):
        """
        Step 5d: Kamiya optimization for bifurcation point

        Optimizes the position of a bifurcation node to minimize total volume
        """
        # Check if this is a bifurcation
        children = list(tree.successors(node))
        parents = list(tree.predecessors(node))

        if len(children) != 2 or len(parents) != 1:
            return  # Not a bifurcation or is root

        parent = parents[0]
        child1, child2 = children

        # Get positions
        p_parent = self.points[parent]
        p_child1 = self.points[child1]
        p_child2 = self.points[child2]
        p_current = self.points[node]

        # Get radii
        r0 = radii[node]  # Parent segment
        r1 = radii[child1]
        r2 = radii[child2]

        # Flow ratios (proportional to L_i)
        f0 = 1.0  # Normalized
        f1 = r1**3  # From Eq. 10: fi ∝ ri³
        f2 = r2**3

        def objective(x):
            """
            Optimize bifurcation position using Kamiya method
            Based on Equations 9-13 from the paper
            """
            # x is the new position of the bifurcation node
            if len(x) == 2:
                p_bif = np.array([x[0], x[1], p_current[2]])  # 2D
            else:
                p_bif = x

            # Compute lengths
            l0 = np.linalg.norm(p_bif - p_parent)
            l1 = np.linalg.norm(p_child1 - p_bif)
            l2 = np.linalg.norm(p_child2 - p_bif)

            if l0 < 1e-6 or l1 < 1e-6 or l2 < 1e-6:
                return np.array([1e10, 1e10])

            # Pressure drop equations (Eq. 9)
            delta1 = f0 * l0 / r0**4 + f1 * l1 / r1**4
            delta2 = f0 * l0 / r0**4 + f2 * l2 / r2**4

            # Murray's law constraint (Eq. 11)
            r0_expected = (f0 * (r1**6/f1 + r2**6/f2))**(1/3)

            # Residuals (Eq. 12)
            residual1 = delta1 * r1**4 - f0 * l0 * r1**4 / r0**4 - f1 * l1
            residual2 = delta2 * r2**4 - f0 * l0 * r2**4 / r0**4 - f2 * l2

            return np.array([residual1, residual2])

        # Initial guess: current position
        x0 = p_current[:2] if len(p_current) == 3 else p_current

        # Bounds: stay within triangle formed by parent and children
        try:
            result = least_squares(objective, x0, method='lm', max_nfev=100)

            if result.success:
                # Update position
                if len(p_current) == 3:
                    new_pos = np.array(
                        [result.x[0], result.x[1], p_current[2]])
                else:
                    new_pos = result.x

                # Check if new position is reasonable (within triangle + margin)
                max_dist = max(np.linalg.norm(p_parent - p_current),
                               np.linalg.norm(p_child1 - p_current),
                               np.linalg.norm(p_child2 - p_current))

                if np.linalg.norm(new_pos - p_current) < 2 * max_dist:
                    self.points[node] = new_pos
                    tree.nodes[node]['pos'] = new_pos

                    # Update edge lengths
                    tree[parent][node]['length'] = np.linalg.norm(
                        new_pos - p_parent)
                    tree[node][child1]['length'] = np.linalg.norm(
                        p_child1 - new_pos)
                    tree[node][child2]['length'] = np.linalg.norm(
                        p_child2 - new_pos)
        except:
            pass  # Keep original position if optimization fails

    def optimize_bifurcation_corrected(self, tree, node, radii):
        """
        Step 5d (fixed): Kamiya optimization for bifurcation point.

        `optimize_bifurcation` above has a bug: its residuals cancel out
        algebraically (delta_i * r_i**4 reduces exactly to the terms being
        subtracted), so the objective is identically ~0 at every position.
        `least_squares` therefore "converges" on the very first evaluation
        without actually moving the bifurcation point.

        This version instead minimizes the true segment-volume objective
        V = f0*l0*r0^2 + f1*l1*r1^2 + f2*l2*r2^2 (with radii held fixed) by
        solving its gradient-equilibrium condition: the weighted sum of unit
        vectors from the bifurcation point to its parent and children,
        scaled by r_i^2, must vanish at the optimum.
        """
        children = list(tree.successors(node))
        parents = list(tree.predecessors(node))

        if len(children) != 2 or len(parents) != 1:
            return  # Not a bifurcation or is root

        parent = parents[0]
        child1, child2 = children

        p_parent = self.points[parent]
        p_child1 = self.points[child1]
        p_child2 = self.points[child2]
        p_current = self.points[node]

        r0 = radii[node]
        r1 = radii[child1]
        r2 = radii[child2]

        f0 = 1.0
        f1 = r1**3
        f2 = r2**3

        def gradient(x):
            if len(x) == 2:
                p_bif = np.array([x[0], x[1], p_current[2]])
            else:
                p_bif = x

            v0 = p_bif - p_parent
            v1 = p_bif - p_child1
            v2 = p_bif - p_child2

            l0 = np.linalg.norm(v0)
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)

            if l0 < 1e-6 or l1 < 1e-6 or l2 < 1e-6:
                return np.full(len(x), 1e10)

            grad = (f0 * r0**2 * v0 / l0 +
                    f1 * r1**2 * v1 / l1 +
                    f2 * r2**2 * v2 / l2)

            return grad[:2] if len(x) == 2 else grad

        x0 = p_current[:2] if len(p_current) == 3 else p_current

        try:
            result = least_squares(gradient, x0, method='lm', max_nfev=100)

            if result.success:
                if len(p_current) == 3:
                    new_pos = np.array(
                        [result.x[0], result.x[1], p_current[2]])
                else:
                    new_pos = result.x

                max_dist = max(np.linalg.norm(p_parent - p_current),
                               np.linalg.norm(p_child1 - p_current),
                               np.linalg.norm(p_child2 - p_current))

                if np.linalg.norm(new_pos - p_current) < 2 * max_dist:
                    self.points[node] = new_pos
                    tree.nodes[node]['pos'] = new_pos

                    tree[parent][node]['length'] = np.linalg.norm(
                        new_pos - p_parent)
                    tree[node][child1]['length'] = np.linalg.norm(
                        p_child1 - new_pos)
                    tree[node][child2]['length'] = np.linalg.norm(
                        p_child2 - new_pos)
        except:
            pass  # Keep original position if optimization fails

    def compute_quality_metrics(self, tree, root, radii, L):
        """
        Step 6: Compute quality metrics for tree evaluation
        """
        metrics = {}

        # 1. Total volume
        total_volume = 0
        for u, v in tree.edges():
            length = tree[u][v]['length']
            radius = radii[v]  # Child radius
            volume = np.pi * radius**2 * length
            total_volume += volume

        metrics['total_volume'] = total_volume

        # 2. Check monotonic radius decrease
        radius_violations = 0
        for node in tree.nodes():
            if node == root:
                continue
            parent = list(tree.predecessors(node))[0]
            if radii[node] > radii[parent]:
                radius_violations += 1

        metrics['radius_violations'] = radius_violations

        # 3. Murray's law compliance at bifurcations
        murray_errors = []
        for node in tree.nodes():
            children = list(tree.successors(node))
            if len(children) == 2:
                r_parent = radii[node]
                r_child1 = radii[children[0]]
                r_child2 = radii[children[1]]

                # Murray's law: r0^γ = r1^γ + r2^γ
                expected = (r_child1**self.gamma + r_child2 **
                            self.gamma)**(1/self.gamma)
                error = abs(r_parent - expected) / r_parent
                murray_errors.append(error)

        metrics['murray_error_mean'] = np.mean(
            murray_errors) if murray_errors else 0
        metrics['murray_error_max'] = np.max(
            murray_errors) if murray_errors else 0

        # 4. Geometric quality - bifurcation angles
        angles = []
        for node in tree.nodes():
            children = list(tree.successors(node))
            if len(children) == 2:
                pos = self.points[node]
                pos1 = self.points[children[0]]
                pos2 = self.points[children[1]]

                v1 = pos1 - pos
                v2 = pos2 - pos

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)
                                              * np.linalg.norm(v2) + 1e-10)
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                angles.append(angle)

        metrics['mean_bifurcation_angle'] = np.mean(angles) if angles else 0
        metrics['min_bifurcation_angle'] = np.min(angles) if angles else 0

        # 5. Root quality - prefer anatomically superior position
        root_z = self.points[root][2] if len(self.points[root]) == 3 else 0
        max_z = np.max(self.points[:, 2]) if len(self.points[0]) == 3 else 1
        metrics['root_z_score'] = root_z / (max_z + 1e-10)

        return metrics

    def reconstruct(self, k_neighbors=10, n_candidates=5, optimize_bifurcations=True,
                    method="highest_z"):
        """
        Main reconstruction pipeline

        Returns:
        --------
        best_tree : networkx.DiGraph
            The best reconstructed tree
        best_root : int
            Index of the best root node
        best_radii : dict
            Radius for each segment
        metrics : dict
            Quality metrics
        """
        # Step 1-2: Build graph
        if self.graph is None:
            self.build_graph_from_skeleton(
                k_neighbors=k_neighbors,
            )

        # Step 3: Terminal points (already identified in graph structure)

        # Step 4: Find root candidates
        candidates = self.find_root_candidates(
            n_candidates=n_candidates, method=method)

        print(f"\nEvaluating {len(candidates)} root candidates...")

        best_score = float('inf')
        best_tree = None
        best_root = None
        best_radii = None
        best_metrics = None

        for i, root in enumerate(candidates):
            print(f"\nCandidate {i+1}/{len(candidates)}: node {root}")

            try:

                # Step 5a: Orient tree from this root
                tree = self.orient_tree_from_root(root)

                # Step 5b: Compute L_i
                L, terminals = self.compute_tree_parameters(tree, root)
                print(
                    f"  Terminals: {len(terminals)}, Total nodes: {tree.number_of_nodes()}")

                # Step 5c: Compute radii
                radii, R, beta, rho = self.compute_radii(tree, root, L)


                # Step 5d: Optimize bifurcations (optional)
                if optimize_bifurcations:
                    bifurcations = [n for n in tree.nodes()
                                    if tree.out_degree(n) == 2 and tree.in_degree(n) == 1]
                    print(f"  Optimizing {len(bifurcations)} bifurcations...")
                    # Limit for speed
                    for step_idx, bif in enumerate(bifurcations[:min(10, len(bifurcations))], start=1):
                        self.optimize_bifurcation(tree, bif, radii)

                    # Recompute radii after optimization
                    radii, R, beta, rho = self.compute_radii(tree, root, L)


                # Step 6: Evaluate quality
                metrics = self.compute_quality_metrics(tree, root, radii, L)

                print(f"  Metrics: volume={metrics['total_volume']:.2f}, "
                      f"murray_error={metrics['murray_error_mean']:.4f}, "
                      f"radius_violations={metrics['radius_violations']}")

                # Scoring: lower is better
                score = (metrics['total_volume'] / 1000 +  # Normalize volume
                         metrics['murray_error_mean'] * 100 +
                         metrics['radius_violations'] * 10 -
                         metrics['root_z_score'] * 50)  # Prefer higher roots

                if score < best_score:
                    best_score = score
                    best_tree = tree
                    best_root = root
                    best_radii = radii
                    best_metrics = metrics


            except Exception as e:
                print(f"  Failed: {e}")
                continue

        if best_tree is None:
            raise RuntimeError("No valid tree found from any root candidate")


        print(f"\n=== Best tree (root={best_root}) ===")
        for k, v in best_metrics.items():
            print(f"  {k}: {v}")

        self.tree_structure = {
            'tree': best_tree,
            'root': best_root,
            'radii': best_radii,
            'metrics': best_metrics
        }

        return best_tree, best_root, best_radii, best_metrics

class MultiTreeReconstruction:
    """
    Reconstruct multiple independent vascular trees from a single point cloud
    """

    def __init__(self, skeleton_points, n_trees=3, gamma=3.0, mu=3.6e-3,
                 Q_perf=0.125, P_out=60, P_in=100):
        """
        Parameters:
        -----------
        skeleton_points : np.array, shape (N, 3)
            3D coordinates of skeleton points
        n_trees : int
            Expected number of separate trees to extract
        gamma, mu, Q_perf, P_out, P_in : float
            CCO physiological parameters (same as VascularTreeReconstruction)
        """
        self.original_points = np.array(skeleton_points)
        self.n_trees = n_trees
        self.gamma = gamma
        self.mu = mu
        self.Q_perf = Q_perf
        self.P_out = P_out
        self.P_in = P_in

        self.trees = []  # List of reconstructed trees
        self.remaining_points = self.original_points.copy()
        self.point_to_tree_mapping = {}  # Maps original point index to tree index

    def reconstruct_multiple_trees(self, k_neighbors_initial=5, k_neighbors_optimization=10,
                                   min_tree_size=50, max_iterations=10,
                                   methods=["highest_z", "highest_z", "lowest_z"]):
        """
        Extract multiple trees iteratively

        Parameters:
        -----------
        k_neighbors_initial : int
            Low k for initial sparse connectivity (helps separate trees)
        k_neighbors_optimization : int
            Higher k for optimizing individual trees
        min_tree_size : int
            Minimum number of points required for a valid tree
        max_iterations : int
            Maximum number of trees to extract

        Returns:
        --------
        trees : list of dict
            List of tree structures, each containing:
            - 'tree': networkx.DiGraph
            - 'root': int (root node in original point indexing)
            - 'radii': dict
            - 'metrics': dict
            - 'points': np.array (points for this tree)
            - 'point_indices': list (indices in original point cloud)
        """
        print(
            f"Starting multi-tree reconstruction from {len(self.original_points)} points")
        print(f"Target: {self.n_trees} trees\n")

        iteration = 0

        while len(self.remaining_points) >= min_tree_size and iteration < max_iterations:
            iteration += 1
            print(f"{'='*60}")
            print(
                f"ITERATION {iteration}: {len(self.remaining_points)} points remaining")
            print(f"{'='*60}\n")

            root_method = methods[(iteration - 1) % len(methods)]
            print(f"Using root candidate method: {root_method}")


            # Extract one tree from remaining points
            tree_data = self._extract_single_tree(
                k_neighbors_initial=k_neighbors_initial,
                k_neighbors_optimization=k_neighbors_optimization,
                min_tree_size=min_tree_size,
                method=root_method,
            )

            if tree_data is None:
                print("No more valid trees found. Stopping.")
                break

            self.trees.append(tree_data)

            # Remove points belonging to this tree
            self._remove_tree_points(tree_data)

            print(
                f"\n✓ Tree {iteration} extracted: {len(tree_data['point_indices'])} points")
            print(f"  Root at: {tree_data['points'][tree_data['root']]}")
            print(
                f"  Terminals: {len([n for n in tree_data['tree'].nodes() if tree_data['tree'].out_degree(n) == 0])}")
            print(
                f"  Total volume: {tree_data['metrics']['total_volume']:.2f} mm³")
            print(f"  Remaining points: {len(self.remaining_points)}\n")

            # Stop if we've extracted enough trees
            if len(self.trees) >= self.n_trees:
                print(
                    f"Extracted target number of trees ({self.n_trees}). Stopping.")
                break

        print(f"\n{'='*60}")
        print(f"FINAL RESULT: {len(self.trees)} trees extracted")
        print(f"{'='*60}")

        for i, tree_data in enumerate(self.trees):
            print(f"Tree {i+1}: {len(tree_data['point_indices'])} points, "
                  f"volume={tree_data['metrics']['total_volume']:.2f} mm³")

        if len(self.remaining_points) > 0:
            print(
                f"\nWarning: {len(self.remaining_points)} points not assigned to any tree")

        return self.trees

    def _extract_single_tree(self, k_neighbors_initial, k_neighbors_optimization,
                             min_tree_size, method):
        """
        Extract one tree from remaining points using sparse connectivity
        """
        if len(self.remaining_points) < min_tree_size:
            return None

        # Step 1: Build sparse graph to identify connected components
        print(f"Step 1: Building sparse graph (k={k_neighbors_initial})...")
        sparse_graph = self._build_sparse_graph(
            self.remaining_points, k_neighbors_initial)

        if sparse_graph.number_of_edges() == 0:
            print("  No edges in sparse graph. Cannot extract tree.")
            return None

        # Step 2: Find largest connected component
        print(f"Step 2: Finding connected components...")
        components = list(nx.connected_components(sparse_graph))
        print(f"  Found {len(components)} components")

        if len(components) == 0:
            return None

        # Get largest component
        largest_component = max(components, key=len)
        print(f"  Largest component: {len(largest_component)} nodes")

        if len(largest_component) < min_tree_size:
            print(f"  Component too small (< {min_tree_size}). Skipping.")
            return None

        # Step 3: Extract points for this component
        component_indices = sorted(list(largest_component))
        component_points = self.remaining_points[component_indices]

        print(
            f"Step 3: Extracting subgraph with {len(component_points)} points...")

        # Step 4: Optimize this tree using standard CCO with higher k
        print(f"Step 4: Optimizing tree (k={k_neighbors_optimization})...")
        reconstructor = VascularTreeReconstruction(
            component_points,
            gamma=self.gamma,
            mu=self.mu,
            Q_perf=self.Q_perf,
            P_out=self.P_out,
            P_in=self.P_in,
        )

        try:
            tree, root, radii, metrics = reconstructor.reconstruct(
                k_neighbors=k_neighbors_optimization,
                n_candidates=5,
                optimize_bifurcations=True,
                method=method,
            )
        except Exception as e:
            print(f"  Failed to reconstruct tree: {e}")
            return None

        # Step 5: Map back to original point indices
        # component_indices maps from component local indices to remaining_points indices
        # We need to map to original point cloud indices

        # Find original indices for these remaining points
        original_indices = self._get_original_indices(component_indices)

        tree_data = {
            'tree': tree,
            'root': root,  # Root in local component indexing
            'radii': radii,
            'metrics': metrics,
            'points': component_points,
            'point_indices': original_indices,  # Indices in original point cloud
            'reconstructor': reconstructor
        }

        return tree_data

    def _build_sparse_graph(self, points, k_neighbors):
        """
        Build a sparse k-NN graph (lower k helps separate disconnected trees)
        """
        from scipy.spatial import cKDTree

        n_points = len(points)
        tree = cKDTree(points)

        # Create graph
        G = nx.Graph()

        # Add all nodes
        for i in range(n_points):
            G.add_node(i)

        # Add k-NN edges
        for i in range(n_points):
            dists, indices = tree.query(
                points[i], k=min(k_neighbors+1, n_points))
            for j, idx in enumerate(indices[1:]):  # Skip self
                if dists[j+1] < np.inf:
                    G.add_edge(i, idx, weight=dists[j+1])

        return G

    def _get_original_indices(self, component_indices):
        """
        Map component indices (in remaining_points) back to original point cloud indices
        """
        # Build reverse mapping: remaining_points → original_points
        original_indices = []

        for comp_idx in component_indices:
            remaining_point = self.remaining_points[comp_idx]

            # Find this point in original cloud
            # Use exact match (or nearest neighbor if points have been modified)
            distances = np.linalg.norm(
                self.original_points - remaining_point, axis=1)
            orig_idx = np.argmin(distances)

            # Verify it's a close match
            if distances[orig_idx] < 0.01:  # Should be exact or very close
                original_indices.append(orig_idx)
            else:
                print(
                    f"Warning: Could not find exact match for point {comp_idx}")
                original_indices.append(orig_idx)  # Use closest anyway

        return original_indices

    def _remove_tree_points(self, tree_data):
        """
        Remove points belonging to extracted tree from remaining points
        """
        # Get indices of points to remove (in remaining_points indexing)
        points_to_remove = tree_data['points']

        # Find which indices in remaining_points to keep
        keep_mask = np.ones(len(self.remaining_points), dtype=bool)

        for i, point in enumerate(self.remaining_points):
            # Check if this point is in the extracted tree
            distances = np.linalg.norm(points_to_remove - point, axis=1)
            if np.min(distances) < 0.01:  # Point is in tree
                keep_mask[i] = False

        # Update remaining points
        self.remaining_points = self.remaining_points[keep_mask]

    def visualize_all_trees(self, show_remaining=True):
        """
        Visualize all extracted trees in a single 3D plot
        """
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Color palette for different trees
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.trees)))

        for tree_idx, tree_data in enumerate(self.trees):
            tree = tree_data['tree']
            points = tree_data['points']
            radii = tree_data['radii']
            root = tree_data['root']
            color = colors[tree_idx]

            # Draw edges
            for u, v in tree.edges():
                pos_u = points[u]
                pos_v = points[v]
                radius = radii[v]
                linewidth = max(0.5, min(5, radius * 10))

                ax.plot([pos_u[0], pos_v[0]],
                        [pos_u[1], pos_v[1]],
                        [pos_u[2], pos_v[2]],
                        color=color, linewidth=linewidth, alpha=0.7)

            # Highlight root
            root_pos = points[root]
            ax.scatter(*root_pos, c=[color], s=200, marker='o',
                       edgecolors='black', linewidths=2,
                       label=f'Tree {tree_idx+1} root')

            # Show terminals
            terminals = [n for n in tree.nodes() if tree.out_degree(n) == 0]
            terminal_pos = points[terminals]
            if len(terminal_pos) > 0:
                ax.scatter(terminal_pos[:, 0], terminal_pos[:, 1], terminal_pos[:, 2],
                           c=[color], s=30, marker='^', alpha=0.5)

        # Show remaining unassigned points
        if show_remaining and len(self.remaining_points) > 0:
            ax.scatter(self.remaining_points[:, 0],
                       self.remaining_points[:, 1],
                       self.remaining_points[:, 2],
                       c='gray', s=5, alpha=0.3, label='Unassigned points')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'Multi-Tree Reconstruction ({len(self.trees)} trees)')

        plt.tight_layout()
        return fig, ax

    def export_all_trees(self, base_filename):
        """
        Export all trees to separate JSON files for viewer

        Parameters:
        -----------
        base_filename : str
            Base name for output files (e.g., "hepatic_vein")
            Will create: hepatic_vein_tree1.json, hepatic_vein_tree2.json, etc.
        """
        import json

        for tree_idx, tree_data in enumerate(self.trees):
            filename = f"{base_filename}_tree{tree_idx+1}.json"

            tree = tree_data['tree']
            points = tree_data['points']
            radii = tree_data['radii']

            branches = []

            for parent, child in tree.edges():
                start_pos = points[parent].tolist()
                end_pos = points[child].tolist()
                radius = float(radii[child])
                Q = float(radius ** 3 * 100)

                branch = {
                    "start": start_pos,
                    "end": end_pos,
                    "radius": radius,
                    "Q": Q
                }
                branches.append(branch)

            with open(filename, 'w') as f:
                json.dump(branches, f, indent=2)

            print(
                f"Tree {tree_idx+1} exported to {filename} ({len(branches)} branches)")

    def get_tree_statistics(self):
        """
        Print summary statistics for all trees
        """
        print(f"\n{'='*60}")
        print(f"MULTI-TREE STATISTICS")
        print(f"{'='*60}\n")

        total_points = sum(len(t['point_indices']) for t in self.trees)
        total_volume = sum(t['metrics']['total_volume'] for t in self.trees)

        print(f"Total trees: {len(self.trees)}")
        print(
            f"Total points assigned: {total_points} / {len(self.original_points)}")
        print(f"Total volume: {total_volume:.2f} mm³")
        print(f"Unassigned points: {len(self.remaining_points)}\n")

        for i, tree_data in enumerate(self.trees):
            print(f"Tree {i+1}:")
            print(f"  Points: {len(tree_data['point_indices'])}")
            print(f"  Root: {tree_data['points'][tree_data['root']]}")

            tree = tree_data['tree']
            terminals = [n for n in tree.nodes() if tree.out_degree(n) == 0]
            bifurcations = [n for n in tree.nodes()
                            if tree.out_degree(n) == 2 and tree.in_degree(n) == 1]

            print(f"  Terminals: {len(terminals)}")
            print(f"  Bifurcations: {len(bifurcations)}")
            print(f"  Volume: {tree_data['metrics']['total_volume']:.2f} mm³")
            print(
                f"  Murray error: {tree_data['metrics']['murray_error_mean']:.4f}")

            radii_values = list(tree_data['radii'].values())
            print(
                f"  Radius range: [{min(radii_values):.3f}, {max(radii_values):.3f}] mm")
            print()


# Helper functions
#
# NOTE: `../vessel_reconstruction/reconstruction.py` is an unreferenced, stale
# (pre-2026-08-20) copy of this module. Nothing imports it; treat this file as
# the only live copy. It is not under version control, so it was left in place
# rather than deleted.

def load_thinning(labels):
    skeleton = skeletonize(labels)
    skeleton_points = np.array(np.where(skeleton)).T
    return skeleton_points


def save_skeleton(skeleton_points, filename):
    # Ensure a pure Python list is saved (JSON cannot serialize numpy arrays)
    to_save = np.array(skeleton_points).tolist()
    with open(filename, 'w') as f:
        json.dump(to_save, f)
    print(f"Skeleton saved to {filename}")


# ---------------------------------------------------------------------------
# Heuristic improvements (all additive -- nothing above this line is modified)
#
# Implements the improvement plan points 1, 2, 3, 5, 6, 7 as a subclass pair so
# the original VascularTreeReconstruction / MultiTreeReconstruction behaviour is
# preserved exactly and can still be run side by side for comparison:
#
#   1. Graph topology from real voxel adjacency instead of Euclidean k-NN + MST
#   2. Radius-aware root candidate selection (needs point_radii from the mask)
#   3. Normalised, weight-based candidate scoring instead of magic constants
#   5. Multifurcations (>2 children) resolved into real binary bifurcations
#   6. Bifurcation optimisation by impact order, multi-pass, in full 3D
#   7. Spur pruning with a data-derived threshold instead of a fixed number
#
# Two latent bugs in the original pipeline are also fixed here, both of which
# silently cancelled most of the effect of any bifurcation optimisation:
#
#   a) `self.points` inherits the dtype of the input. Skeleton points loaded
#      from JSON/`np.where` are integers, so `self.points[node] = new_pos`
#      truncated every optimiser move back to integer voxel coordinates.
#      Fixed by casting the point cloud to float.
#   b) `MultiTreeReconstruction._extract_single_tree` stored the *pre*-optimisation
#      copy of the points in `tree_data['points']`, which is what
#      `export_all_trees` / `visualize_all_trees` read -- so optimised positions
#      never reached the viewer. Fixed by exporting `reconstructor.points`.
# ---------------------------------------------------------------------------


def estimate_point_radii_from_mask(mask, points, spacing=None):
    """
    Estimate a local vessel radius for each skeleton point (improvement plan #2).

    Uses the Euclidean distance transform of the binary vessel mask: at a
    centreline voxel, the distance to the nearest background voxel is the radius
    of the largest inscribed sphere, i.e. the local vessel radius.

    Parameters:
    -----------
    mask : np.ndarray
        Binary (or truthy) vessel mask, same volume the skeleton came from.
    points : np.array, shape (N, 3)
        Skeleton point coordinates as voxel indices into `mask`.
    spacing : tuple or None
        Optional physical voxel spacing passed to the distance transform; with
        None the radii come out in voxel units.

    Returns:
    --------
    np.array, shape (N,) of local radii.
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)

    edt = distance_transform_edt(mask, sampling=spacing)

    idx = np.rint(np.asarray(points)).astype(int)
    for d in range(idx.shape[1]):
        idx[:, d] = np.clip(idx[:, d], 0, mask.shape[d] - 1)

    return edt[tuple(idx.T)]


class ImprovedVascularTreeReconstruction(VascularTreeReconstruction):
    """
    VascularTreeReconstruction with the heuristic improvements applied.

    Every override keeps the base-class signature, so an instance can be dropped
    in wherever the original is used. The original methods remain reachable via
    `super()` (and unchanged on the parent class) for A/B comparison.

    Not a standalone pipeline choice: it is kept because
    `LabelConnectedMultiTreeReconstruction` uses it for the per-group tree
    maths -- see `README_LABEL_CONNECTED.md`
    §9 and `lab_diary.md`'s session log). Kept in this file only because it's
    a hard dependency of two things that were *not* pruned:
    `LabelConnectedMultiTreeReconstruction` (uses it for all per-group
    reconstruction) and `reconstruction_fixed.py`'s
    `FixedVascularTreeReconstruction` (subclasses it directly). Do not
    reintroduce it as a user-facing pipeline option without checking those
    dependents still work.
    """

    DEFAULT_SCORE_WEIGHTS = {
        'total_volume': 1.0,
        'murray_error_mean': 1.0,
        'radius_violations': 1.0,
        'root_z': 0.5,
    }

    def __init__(self, skeleton_points, gamma=3.0, mu=3.6e-3,
                 Q_perf=0.125, P_out=60, P_in=100,
                 point_radii=None,
                 adjacency_radius=1.0,
                 bridge_components=True,
                 max_bridge_length=None,
                 bridge_factor=5.0,
                 auto_prune=False,
                 prune_factor=1.5,
                 prune_max_passes=1,
                 max_spur_nodes=2,
                 resolve_multifurcations=True,
                 bifurcation_passes=3,
                 max_bifurcations=None,
                 optimize_in_3d=True,
                 score_weights=None):
        """
        Extra parameters (all with conservative defaults):
        --------------------------------------------------
        point_radii : np.array (N,) or None
            Local vessel radius per skeleton point, e.g. from
            `estimate_point_radii_from_mask`. Enables the "thickest" root
            selection methods; without it those fall back to "highest_z".
        adjacency_radius : float
            Chebyshev radius for voxel adjacency. 1.0 is the 26-neighbourhood.
        bridge_components / max_bridge_length / bridge_factor :
            Whether to reconnect components that voxel adjacency leaves apart,
            and how far a bridge may span. `max_bridge_length=None` means *no
            limit*: every component is reconnected, matching the original
            pipeline's global MST, so no point is ever stranded outside the
            root's component. Pass a number (or `bridge_factor * median edge
            length` explicitly) only if you deliberately want gaps left open.
        auto_prune / prune_factor / prune_max_passes / max_spur_nodes :
            Data-derived spur pruning (#7), and **off by default**: the original
            pipeline never prunes unless asked (`max_edge_length=None`), and for
            vascular reconstruction discarding peripheral branches loses exactly
            the fine structure we are trying to recover. When enabled the
            defaults are deliberately conservative -- one pass, threshold
            `prune_factor * median(edge length)`, and a spur is only removed if
            it is at most `max_spur_nodes` nodes long -- so it strips single
            voxel spikes rather than eating branches tip-first over many passes.
        resolve_multifurcations : bool
            Split >2-child nodes into binary bifurcations (#5).
        bifurcation_passes / max_bifurcations / optimize_in_3d :
            Bifurcation optimisation control (#6). `max_bifurcations=None`
            means all of them, instead of the original hardcoded first 10.
        score_weights : dict or None
            Weights for the normalised candidate score (#3).
        """
        super().__init__(skeleton_points, gamma=gamma, mu=mu, Q_perf=Q_perf,
                         P_out=P_out, P_in=P_in,
                         )

        # Latent bug (a): integer skeleton coordinates silently truncate every
        # optimiser move. Work in floating point.
        self.points = np.asarray(self.points, dtype=float)

        self.point_radii = (np.asarray(point_radii, dtype=float)
                            if point_radii is not None else None)

        self.adjacency_radius = adjacency_radius
        self.bridge_components = bridge_components
        self.max_bridge_length = max_bridge_length
        self.bridge_factor = bridge_factor

        self.auto_prune = auto_prune
        self.prune_factor = prune_factor
        self.prune_max_passes = prune_max_passes
        self.max_spur_nodes = max_spur_nodes

        self.resolve_multifurcations_enabled = resolve_multifurcations
        self.bifurcation_passes = bifurcation_passes
        self.max_bifurcations = max_bifurcations
        self.optimize_in_3d = optimize_in_3d

        self.score_weights = dict(score_weights) if score_weights else dict(
            self.DEFAULT_SCORE_WEIGHTS)

        # Filled in by build_graph_from_skeleton for inspection/plotting
        self.graph_stats = {}

    # -- Improvement #1: topology from voxel adjacency ----------------------

    def build_graph_from_skeleton(self, k_neighbors=10, max_edge_length=None):
        """
        Build the graph from real skeleton voxel adjacency (improvement #1).

        The original builds a k-NN graph on Euclidean distance and takes its MST,
        which happily connects two points that are close in space but belong to
        different vessels (parallel branches, a vessel folding back on itself).
        Skeletonisation already tells us which voxels are actually connected, so
        this uses the 26-neighbourhood instead, then:
          * breaks any cycles skeletonisation left behind (MST per component),
          * optionally bridges genuinely separate components, but only across
            gaps no longer than `max_bridge_length`.

        `k_neighbors` is accepted and ignored -- kept so this stays signature
        compatible with the base method. `max_edge_length`, as in the base
        class, is used as the *minimum* branch length for pruning (the base
        class's own naming); None means "derive the threshold from the data".
        """
        print("Building graph from skeleton voxel adjacency...")
        pts = self.points
        n_points = len(pts)

        G = nx.Graph()
        for i, pos in enumerate(pts):
            G.add_node(i, pos=pos)

        # Chebyshev radius 1 over integer voxel coordinates == 26-neighbourhood
        kdt = cKDTree(pts)
        pairs = kdt.query_pairs(r=self.adjacency_radius, p=np.inf)
        for i, j in pairs:
            d = float(np.linalg.norm(pts[i] - pts[j]))
            G.add_edge(int(i), int(j), length=d)

        n_adjacency_edges = G.number_of_edges()
        n_components_raw = nx.number_connected_components(G)

        # Skeletonisation can leave small loops; keep connectivity, drop cycles.
        forest = nx.Graph()
        forest.add_nodes_from(G.nodes(data=True))
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            if sub.number_of_edges() == 0:
                continue
            forest.add_edges_from(
                nx.minimum_spanning_tree(sub, weight='length').edges(data=True))
        n_cycle_edges = n_adjacency_edges - forest.number_of_edges()

        self.graph = forest
        print(f"  Adjacency edges: {n_adjacency_edges} "
              f"({n_cycle_edges} cycle edges removed)")
        print(f"  Connected components from adjacency: {n_components_raw}")

        n_bridges = 0
        if self.bridge_components:
            n_bridges = self._bridge_components()
            print(f"  Bridged {n_bridges} gaps between components -> "
                  f"{nx.number_connected_components(self.graph)} components")


        print(f"Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

        n_pruned = 0
        if self.auto_prune or max_edge_length is not None:
            n_pruned = self.prune_spurs_adaptive(threshold=max_edge_length)

        n_components_final = nx.number_connected_components(self.graph)
        self.graph_stats = {
            'n_points': n_points,
            'n_adjacency_edges': n_adjacency_edges,
            'n_cycle_edges_removed': n_cycle_edges,
            'n_components_raw': n_components_raw,
            'n_bridges': n_bridges,
            'n_pruned_nodes': n_pruned,
            'n_nodes_final': self.graph.number_of_nodes(),
            'n_edges_final': self.graph.number_of_edges(),
            'n_components_final': n_components_final,
            'node_retention': self.graph.number_of_nodes() / max(n_points, 1),
        }

        # Anything not in the root's component is dropped by
        # `orient_tree_from_root`, so warn loudly rather than losing vessel.
        if n_components_final > 1:
            print(f"  WARNING: graph still has {n_components_final} components; "
                  f"only the root's component will be reconstructed")

        return self.graph

    def _bridge_components(self, max_bridge_length=None):
        """
        Reconnect components left separate by voxel adjacency, shortest first.

        Builds a component-level graph whose edge weights are the true minimum
        point-to-point distance between the two components, takes its MST, and
        materialises those bridges that are short enough. Unlike a global
        Euclidean MST this can only add a bridge where there is a real gap, and
        never re-routes an already-adjacent stretch of vessel.
        """
        G = self.graph
        comps = [sorted(c) for c in nx.connected_components(G)]
        if len(comps) <= 1:
            return 0

        if max_bridge_length is None:
            max_bridge_length = self.max_bridge_length
        if max_bridge_length is None:
            # No limit: reconnect everything. `orient_tree_from_root` keeps only
            # the root's connected component, so any component left unbridged is
            # silently dropped from the reconstruction -- for vascular data that
            # means losing real vessel, so gaps are closed by default.
            max_bridge_length = np.inf

        comp_trees = [cKDTree(self.points[c]) for c in comps]

        CG = nx.Graph()
        CG.add_nodes_from(range(len(comps)))
        for a in range(len(comps)):
            pts_a = self.points[comps[a]]
            for b in range(a + 1, len(comps)):
                dists, idxs = comp_trees[b].query(pts_a, k=1)
                amin = int(np.argmin(dists))
                CG.add_edge(a, b,
                            weight=float(dists[amin]),
                            u=int(comps[a][amin]),
                            v=int(comps[b][int(idxs[amin])]))

        added = 0
        for _, _, data in nx.minimum_spanning_tree(CG, weight='weight').edges(data=True):
            if data['weight'] <= max_bridge_length:
                G.add_edge(data['u'], data['v'],
                           length=float(data['weight']), bridge=True)
                added += 1

        return added

    # -- Improvement #7: data-derived spur pruning --------------------------

    def spur_paths(self):
        """
        Find every terminal spur: the run of nodes from a degree-1 endpoint up to
        (but excluding) the first real junction.

        Returns a list of (total_length, [nodes to remove], junction_node).
        Spurs whose walk reaches another endpoint instead of a junction are
        skipped -- that means the whole component is a simple path, and pruning
        it would delete a vessel rather than a noise spike.
        """
        G = self.graph
        spurs = []

        for endpoint in [n for n in G.nodes() if G.degree(n) == 1]:
            path = [endpoint]
            total = 0.0
            prev = endpoint
            neighbours = list(G.neighbors(endpoint))
            if not neighbours:
                continue
            cur = neighbours[0]

            while True:
                total += G[prev][cur]['length']
                degree = G.degree(cur)

                if degree >= 3:
                    spurs.append((total, path[:], cur))
                    break
                if degree == 1:
                    break  # simple-path component: leave it alone

                path.append(cur)
                onward = [x for x in G.neighbors(cur) if x != prev]
                if not onward:
                    break
                prev, cur = cur, onward[0]

        return spurs

    def prune_spurs_adaptive(self, threshold=None, factor=None, max_passes=None,
                             max_spur_nodes=None, verbose=True):
        """
        Prune short terminal branches with a data-derived threshold (#7).

        The original `_prune_branches` compares a single *edge* against a fixed
        absolute `min_length`, so the same number over-prunes one dataset and
        under-prunes another, and it can raise IndexError when two degree-1
        nodes are each other's only neighbour. This measures the whole spur and
        derives the threshold from the graph itself
        (`factor * median(edge length)`), so it scales with voxel spacing and
        sampling density automatically.
        """
        if factor is None:
            factor = self.prune_factor
        if max_passes is None:
            max_passes = self.prune_max_passes
        if max_spur_nodes is None:
            max_spur_nodes = self.max_spur_nodes

        lengths = [d['length'] for _, _, d in self.graph.edges(data=True)]
        if not lengths:
            return 0

        if threshold is None:
            threshold = factor * float(np.median(lengths))

        removed_total = 0
        for _ in range(max_passes):
            to_remove = set()
            for total, path, _junction in self.spur_paths():
                # Length alone is not enough: repeated passes would nibble a
                # real branch away one tip at a time. Only remove genuinely
                # tiny spikes.
                if total < threshold and len(path) <= max_spur_nodes:
                    to_remove.update(path)
            if not to_remove:
                break
            self.graph.remove_nodes_from(to_remove)
            removed_total += len(to_remove)

        if verbose:
            print(f"  Adaptive pruning (threshold={threshold:.3f}): "
                  f"removed {removed_total} nodes -> "
                  f"{self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges")

        return removed_total

    # -- Improvement #2: radius-aware root selection ------------------------

    def find_root_candidates(self, n_candidates=5, method="highest_z"):
        """
        Root candidates, with radius-aware strategies added (improvement #2).

        New methods on top of the base "highest_z"/"lowest_z"/"centrality":
          * "thickest"   -- endpoints with the largest local vessel radius. The
                            root of a vascular tree is essentially always its
                            widest segment, and that signal was being discarded.
          * "thickest_z" -- normalised radius and height combined equally, which
                            is the usual hepatic-vein prior (wide *and* superior,
                            i.e. towards the IVC).
        Both fall back to the base implementation if no `point_radii` were given.
        """
        if method not in ("thickest", "thickest_z"):
            return super().find_root_candidates(n_candidates=n_candidates,
                                                method=method)

        if self.point_radii is None:
            print("  No point_radii available; falling back to 'highest_z'")
            return super().find_root_candidates(n_candidates=n_candidates,
                                                method="highest_z")

        endpoints = [n for n in self.graph.nodes() if self.graph.degree(n) == 1]
        if len(endpoints) == 0:
            degrees = dict(self.graph.degree())
            endpoints = sorted(degrees, key=degrees.get, reverse=True)[:10]

        if len(endpoints) == 0:
            print("  No candidate nodes in graph")
            return []

        radii = np.array([self.point_radii[n] if n < len(self.point_radii)
                          else 0.0 for n in endpoints], dtype=float)

        if method == "thickest":
            score = radii
        else:
            z = np.array([self.points[n][2] for n in endpoints], dtype=float)
            score = 0.5 * self._minmax(radii) + 0.5 * self._minmax(z)

        order = np.argsort(score)[::-1][:n_candidates]
        candidates = [endpoints[int(i)] for i in order]

        print(f"Found {len(candidates)} root candidates by '{method}' "
              f"(radius range {radii.min():.2f}-{radii.max():.2f})")
        return candidates

    # -- Improvement #4 (already implemented) reused here -------------------

    def compute_radii(self, tree, root, L):
        """Use the iterative terminal-resistance solve instead of R=1.0."""
        return self.compute_radii_iterative(tree, root, L)

    # -- Improvement #5: resolve multifurcations ----------------------------

    def resolve_multifurcations(self, tree, root, offset_fraction=0.25):
        """
        Split nodes with >2 children into a chain of real bifurcations (#5).

        The CCO radius formulas only define a branching ratio for a binary
        split; the original code gives every child of a trifurcation
        `beta = 1.0`, silently abandoning Murray's law exactly where the
        skeleton topology is least trustworthy. Here the two children whose
        directions are most aligned are repeatedly merged behind a new short
        intermediate segment until every node is binary.

        The new nodes are appended to `self.points`, so node ids stay usable as
        indices everywhere else in the pipeline.
        """
        added = 0

        for node in list(tree.nodes()):
            children = list(tree.successors(node))

            while len(children) > 2:
                pos = self.points[node]

                directions = {}
                for c in children:
                    v = self.points[c] - pos
                    norm = float(np.linalg.norm(v))
                    directions[c] = v / norm if norm > 1e-12 else v

                best_pair, best_cos = None, -2.0
                for a in range(len(children)):
                    for b in range(a + 1, len(children)):
                        ca, cb = children[a], children[b]
                        cos = float(np.dot(directions[ca], directions[cb]))
                        if cos > best_cos:
                            best_cos, best_pair = cos, (ca, cb)

                c1, c2 = best_pair

                mean_dir = directions[c1] + directions[c2]
                norm = float(np.linalg.norm(mean_dir))
                if norm < 1e-12:
                    mean_dir, norm = directions[c1], 1.0
                mean_dir = mean_dir / norm

                d1 = float(np.linalg.norm(self.points[c1] - pos))
                d2 = float(np.linalg.norm(self.points[c2] - pos))
                step = offset_fraction * max(min(d1, d2), 1e-6)
                new_pos = pos + step * mean_dir

                # Append immediately so the id is valid for the next iteration
                self.points = np.vstack([self.points, new_pos.reshape(1, -1)])
                new_id = int(self.points.shape[0] - 1)

                tree.add_node(new_id, pos=new_pos)
                tree.add_edge(node, new_id,
                              length=float(np.linalg.norm(new_pos - pos)))
                for c in (c1, c2):
                    tree.remove_edge(node, c)
                    tree.add_edge(new_id, c,
                                  length=float(np.linalg.norm(self.points[c] - new_pos)))

                children = list(tree.successors(node))
                added += 1

        return added

    # -- Improvement #6: better bifurcation optimisation --------------------

    def optimize_bifurcation(self, tree, node, radii):
        """
        Corrected Kamiya optimisation, in full 3D (improvement #6).

        Same volume-equilibrium objective as `optimize_bifurcation_corrected`
        (the parent's fixed version), but solving for all three coordinates
        instead of holding z fixed -- hepatic veins branch craniocaudally, so
        pinning z discards a real degree of freedom.
        """
        children = list(tree.successors(node))
        parents = list(tree.predecessors(node))

        if len(children) != 2 or len(parents) != 1:
            return

        parent = parents[0]
        child1, child2 = children

        p_parent = self.points[parent]
        p_child1 = self.points[child1]
        p_child2 = self.points[child2]
        p_current = self.points[node]

        r0 = radii[node]
        r1 = radii[child1]
        r2 = radii[child2]

        f0, f1, f2 = 1.0, r1**3, r2**3

        def gradient(x):
            p_bif = np.asarray(x, dtype=float)

            v0 = p_bif - p_parent
            v1 = p_bif - p_child1
            v2 = p_bif - p_child2

            l0 = np.linalg.norm(v0)
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)

            if l0 < 1e-6 or l1 < 1e-6 or l2 < 1e-6:
                return np.full(len(x), 1e10)

            return (f0 * r0**2 * v0 / l0 +
                    f1 * r1**2 * v1 / l1 +
                    f2 * r2**2 * v2 / l2)

        x0 = np.asarray(p_current, dtype=float)
        if not self.optimize_in_3d:
            return self.optimize_bifurcation_corrected(tree, node, radii)

        try:
            result = least_squares(gradient, x0, method='lm', max_nfev=200)

            if result.success:
                new_pos = np.asarray(result.x, dtype=float)

                max_dist = max(np.linalg.norm(p_parent - p_current),
                               np.linalg.norm(p_child1 - p_current),
                               np.linalg.norm(p_child2 - p_current))

                if np.linalg.norm(new_pos - p_current) < 2 * max_dist:
                    self.points[node] = new_pos
                    tree.nodes[node]['pos'] = new_pos

                    tree[parent][node]['length'] = float(
                        np.linalg.norm(new_pos - p_parent))
                    tree[node][child1]['length'] = float(
                        np.linalg.norm(p_child1 - new_pos))
                    tree[node][child2]['length'] = float(
                        np.linalg.norm(p_child2 - new_pos))
        except Exception:
            pass  # keep original position if optimisation fails

    def optimize_bifurcations_by_impact(self, tree, radii, L, root,
                                        max_bifurcations=None, passes=None):
        """
        Optimise bifurcations largest-subtree-first, over several passes (#6).

        The original takes `bifurcations[:10]` in whatever order the nodes
        happen to come out of the graph -- an arbitrary 10 of (here) 70, with no
        relation to which ones matter. Ordering by the number of terminals fed
        by each node puts the trunk bifurcations first, and repeating the sweep
        lets neighbouring bifurcations settle against each other, since moving
        one changes the optimum for the next.
        """
        if max_bifurcations is None:
            max_bifurcations = self.max_bifurcations
        if passes is None:
            passes = self.bifurcation_passes

        bifurcations = [n for n in tree.nodes()
                        if tree.out_degree(n) == 2 and tree.in_degree(n) == 1]
        bifurcations.sort(
            key=lambda n: (L.get(n, 0), radii.get(n, 0.0)), reverse=True)

        if max_bifurcations is not None:
            bifurcations = bifurcations[:max_bifurcations]

        for _ in range(passes):
            for bif in bifurcations:
                self.optimize_bifurcation(tree, bif, radii)
            radii, _R, _beta, _rho = self.compute_radii(tree, root, L)

        return radii

    # -- Improvement #3: normalised candidate scoring -----------------------

    @staticmethod
    def _minmax(values):
        """Scale to [0, 1]; all-equal input maps to all zeros."""
        v = np.asarray(values, dtype=float)
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi - lo < 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    def score_candidates(self, metrics_list, method="highest_z", weights=None):
        """
        Score root candidates on normalised criteria (improvement #3).

        The original score

            volume/1000 + murray*100 + violations*10 - root_z*50

        mixes mm3, a dimensionless error, a count and a ratio through magic
        constants, so whichever term happens to have the widest numeric range
        decides the winner by accident. Worse, the `- root_z * 50` term always
        rewards high roots, which directly fights the "lowest_z" strategy the
        caller asked for.

        Here each criterion is min-max normalised across the candidates being
        compared, so the weights are actually commensurate, and the height term
        follows the requested `method` instead of overriding it. Lower is better.
        """
        weights = dict(self.score_weights if weights is None else weights)

        volume = self._minmax([m['total_volume'] for m in metrics_list])
        murray = self._minmax([m['murray_error_mean'] for m in metrics_list])
        violations = self._minmax([m['radius_violations'] for m in metrics_list])
        z = self._minmax([m['root_z_score'] for m in metrics_list])

        if method == "lowest_z":
            z_penalty = z              # high roots are the bad ones here
        elif method in ("highest_z", "thickest_z"):
            z_penalty = 1.0 - z
        else:
            z_penalty = np.zeros_like(z)

        return (weights.get('total_volume', 1.0) * volume +
                weights.get('murray_error_mean', 1.0) * murray +
                weights.get('radius_violations', 1.0) * violations +
                weights.get('root_z', 0.5) * z_penalty)

    # -- Pipeline ----------------------------------------------------------

    def reconstruct(self, k_neighbors=10, n_candidates=5,
                    optimize_bifurcations=True, method="highest_z"):
        """
        Reconstruction pipeline with improvements #1, #2, #3, #4, #5, #6, #7.

        Structurally the same as the base pipeline, with three behavioural
        differences beyond the individual improvements:
          * every candidate is evaluated before any is chosen, because the
            normalised score needs the whole field to normalise against;
          * each candidate starts from a pristine copy of the point cloud, so
            one candidate's bifurcation moves can't leak into the next;
          * the recursion limit is raised for the depth-recursive radius solve,
            since voxel-adjacency chains are far longer than k-NN/MST ones.
        """
        import sys

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 20000))

        try:
            if self.graph is None:
                self.build_graph_from_skeleton(
                    k_neighbors=k_neighbors)


            # `orient_tree_from_root` only ever keeps the root's connected
            # component, so a root landing in a small fragment reconstructs a
            # stub and silently discards the rest of the vessel. Restrict the
            # candidate search to the largest component instead; whatever is
            # left over is reported as uncovered so the caller can return it
            # to the point pool for the next iteration rather than throw it away.
            n_components = nx.number_connected_components(self.graph)
            if n_components > 1:
                largest = max(nx.connected_components(self.graph), key=len)
                print(f"  Graph has {n_components} components; restricting to "
                      f"the largest ({len(largest)}/{self.graph.number_of_nodes()} nodes)")
                self.graph = self.graph.subgraph(largest).copy()

            candidates = self.find_root_candidates(
                n_candidates=n_candidates, method=method)

            print(f"\nEvaluating {len(candidates)} root candidates...")

            points_backup = self.points.copy()
            results = []

            for i, root in enumerate(candidates):
                print(f"\nCandidate {i+1}/{len(candidates)}: node {root}")

                # Independent starting geometry per candidate
                self.points = points_backup.copy()

                try:
                    tree = self.orient_tree_from_root(root)

                    n_split = 0
                    if self.resolve_multifurcations_enabled:
                        n_split = self.resolve_multifurcations(tree, root)
                        if n_split:
                            print(f"  Resolved {n_split} multifurcations")

                    L, terminals = self.compute_tree_parameters(tree, root)
                    print(f"  Terminals: {len(terminals)}, "
                          f"Total nodes: {tree.number_of_nodes()}")

                    radii, R, beta, rho = self.compute_radii(tree, root, L)

                    if optimize_bifurcations:
                        n_bifs = len([n for n in tree.nodes()
                                      if tree.out_degree(n) == 2
                                      and tree.in_degree(n) == 1])
                        print(f"  Optimizing {n_bifs} bifurcations "
                              f"({self.bifurcation_passes} passes)...")
                        radii = self.optimize_bifurcations_by_impact(
                            tree, radii, L, root)

                    metrics = self.compute_quality_metrics(tree, root, radii, L)

                    print(f"  Metrics: volume={metrics['total_volume']:.2f}, "
                          f"murray_error={metrics['murray_error_mean']:.4f}, "
                          f"radius_violations={metrics['radius_violations']}")

                    results.append({
                        'tree': tree,
                        'root': root,
                        'radii': radii,
                        'metrics': metrics,
                        'points': self.points.copy(),
                        'n_multifurcations_split': n_split,
                    })

                except Exception as e:
                    print(f"  Failed: {e}")
                    continue

            self.points = points_backup

            if not results:
                raise RuntimeError("No valid tree found from any root candidate")

            scores = self.score_candidates(
                [r['metrics'] for r in results], method=method)
            best_idx = int(np.argmin(scores))
            best = results[best_idx]

            # Adopt the winning candidate's geometry (optimised positions and
            # any nodes added while resolving multifurcations).
            self.points = best['points']

            print(f"\n=== Best tree (root={best['root']}, "
                  f"score={scores[best_idx]:.4f}) ===")
            for k, v in best['metrics'].items():
                print(f"  {k}: {v}")


            self.tree_structure = {
                'tree': best['tree'],
                'root': best['root'],
                'radii': best['radii'],
                'metrics': best['metrics'],
                'all_scores': scores,
            }

            return best['tree'], best['root'], best['radii'], best['metrics']

        finally:
            sys.setrecursionlimit(old_limit)


# ---------------------------------------------------------------------------
# Label-continuity-driven multi-tree extraction
#
# Everything above builds connectivity either from raw Euclidean proximity
# (MultiTreeReconstruction) or from voxel adjacency with gaps bridged shut
# (`ImprovedVascularTreeReconstruction` with `bridge_components=True`, so a
# fragmented label mask still reconstructs as one tree -- see the point-loss
# bug in the lab diary). This class does the opposite on purpose: it never fabricates a
# connection the label data doesn't actually contain, using the Zenodo
# Task08_HepaticVessel "_mod" masks (manually cleaned so each anatomical vein
# is, as much as possible, one continuous label region) as the source of truth
# for what counts as "connected".
# ---------------------------------------------------------------------------


class LabelConnectedMultiTreeReconstruction(MultiTreeReconstruction):
    """
    Multi-tree extraction driven by mask/label continuity instead of Euclidean
    proximity or gap-bridging.

    Two stages, matching how the task was posed:

    Stage 1 -- "reconnect only the parts where the labels are continuous": build
    a graph using real voxel adjacency only (26-neighbourhood / Chebyshev radius
    1), with **no** bridging across gaps at all -- the opposite default of
    `ImprovedVascularTreeReconstruction`. Each natural connected component of
    that graph is a tree candidate, largest first.

    Stage 2 -- "then try to get to 3 trees": hepatic veins usually join into a
    common trunk near the IVC, so stage 1 alone typically produces ONE dominant
    continuous component rather than three separate ones. When fewer than
    `n_trees` usable natural components exist, the largest is split into
    multiple subtrees.

    Two split methods (`split_method`, default `"bifurcation"`):

    - `"bifurcation"` (default, `_split_by_bifurcation`) -- cut the dominant
      component's own tree (already built cycle-free in stage 1) at its real
      branch points: nodes of degree >= 3 are genuine forks the segmentation
      itself shows, not invented. The `n_trees - 1` largest child branches
      (ranked by summed `radius**3`, a Murray's-law-consistent size proxy) are
      cut off as their own trees; if there aren't enough real forks to reach
      `n_trees`, the remainder is found by cutting the thinnest edge along the
      trunk's own diameter path (a physical narrowing, not a geometric
      midpoint) as a fallback; if even that runs out, fewer than `n_trees`
      trees are returned rather than forcing a boundary that isn't there.
    - `"voronoi"` (the original method, `_select_split_seeds` +
      `_voronoi_partition`) -- multi-source shortest-path partition (a
      discrete Voronoi diagram over the voxel-adjacency graph), seeded at
      well-separated, radius-preferring candidate points. Purely geometric:
      it does not know where a real vessel boundary is, so a boundary can
      fall through the middle of a straight, undivided vessel segment. Kept
      for comparison; not the default as of 2026-08-24 -- see
      `README_LABEL_CONNECTED.md` section 6.2/9 for why.

    Every resulting point group -- natural component or Voronoi partition alike
    -- is hydraulically reconstructed by the *same*
    `ImprovedVascularTreeReconstruction.reconstruct()` pipeline already used
    elsewhere in this module (radius-aware roots, iterative terminal-resistance
    radii, impact-ordered 3D bifurcation optimisation, normalised candidate
    scoring). No new reconstruction math is introduced here -- only how points
    get grouped into trees changes.

    Subclasses `MultiTreeReconstruction` purely to reuse its output-side methods
    (`visualize_all_trees`, `export_all_trees`, `get_tree_statistics`,
    `_get_original_indices`), which only ever read `self.trees` /
    `self.remaining_points` / `self.original_points` and work regardless of how
    `self.trees` was populated. `reconstruct_multiple_trees` is fully replaced.
    """

    def __init__(self, skeleton_points, mask=None, n_trees=3, gamma=3.0, mu=3.6e-3,
                 Q_perf=0.125, P_out=60, P_in=100,
                 point_radii=None, adjacency_radius=1.0, min_tree_size=10,
                 builder_kwargs=None, split_method="bifurcation"):
        """
        Extra parameters beyond MultiTreeReconstruction:
        -------------------------------------------------
        mask : np.ndarray or None
            The binary label volume the skeleton was thinned from (e.g. a
            Task08_HepaticVessel `*_mod.nii.gz`). Used only to derive
            `point_radii` via `estimate_point_radii_from_mask` when
            `point_radii` isn't given directly; the mask's own connectivity is
            *not* re-derived here since we already have a connected skeleton.
        point_radii : np.array (N,) or None
            Local vessel radius per skeleton point. If omitted and `mask` is
            given, computed automatically. Drives split-seed selection and the
            "thickest"/"thickest_z" root strategies inside each tree's own
            `reconstruct()` call.
        adjacency_radius : float
            Chebyshev radius defining "continuous" for stage 1. 1.0 is the
            26-neighbourhood, matching `ImprovedVascularTreeReconstruction`.
        min_tree_size : int
            Components/partitions smaller than this are never turned into a
            tree, matching `MultiTreeReconstruction`'s own parameter.
        builder_kwargs : dict or None
            Forwarded to every `ImprovedVascularTreeReconstruction` used for
            the actual per-tree reconstruction.
        split_method : "bifurcation" (default) or "voronoi"
            How stage 2 splits the dominant component when stage 1 alone
            doesn't reach `n_trees`. See the class docstring's Stage 2
            section for what each does and why the default changed
            2026-08-24.
        """
        super().__init__(skeleton_points, n_trees=n_trees, gamma=gamma, mu=mu,
                         Q_perf=Q_perf, P_out=P_out, P_in=P_in,
                         )

        self.original_points = np.asarray(self.original_points, dtype=float)
        self.remaining_points = self.original_points.copy()

        self.mask = mask
        self.adjacency_radius = adjacency_radius
        self.min_tree_size = min_tree_size
        self.builder_kwargs = dict(builder_kwargs or {})
        if split_method not in ("bifurcation", "voronoi"):
            raise ValueError(f"split_method must be 'bifurcation' or 'voronoi', got {split_method!r}")
        self.split_method = split_method

        if point_radii is not None:
            self.point_radii = np.asarray(point_radii, dtype=float)
        elif mask is not None:
            self.point_radii = estimate_point_radii_from_mask(
                mask, self.original_points)
        else:
            self.point_radii = None

        self.label_graph = None
        self.component_sizes = []

    # -- Stage 1: strict, unbridged connectivity -----------------------------

    def _build_label_graph(self):
        """
        Voxel-adjacency graph over the *entire* input point cloud, with cycles
        (from skeletonisation loops) removed per component but, critically, no
        `_bridge_components` step at all: two points that are not actually
        adjacent in the label data stay in separate components, full stop.

        Returns connected components as lists of point indices, largest first.
        """
        pts = self.original_points
        G = nx.Graph()
        G.add_nodes_from(range(len(pts)))

        kdt = cKDTree(pts)
        for i, j in kdt.query_pairs(r=self.adjacency_radius, p=np.inf):
            length = float(np.linalg.norm(pts[i] - pts[j]))
            G.add_edge(int(i), int(j), length=length)

        forest = nx.Graph()
        forest.add_nodes_from(G.nodes(data=True))
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            if sub.number_of_edges() == 0:
                continue
            forest.add_edges_from(
                nx.minimum_spanning_tree(sub, weight='length').edges(data=True))

        self.label_graph = forest

        components = sorted(
            (sorted(c) for c in nx.connected_components(forest)),
            key=len, reverse=True)
        self.component_sizes = [len(c) for c in components]

        return components

    # -- Stage 2: splitting the dominant component ---------------------------

    def _select_split_seeds(self, nodes, n_seeds):
        """
        Farthest-point sampling (by graph distance) over one component, to pick
        `n_seeds` well-separated candidate roots for the Voronoi split. The
        first seed is the thickest point available (falls back to an arbitrary
        node if no radii are known), so at least one seed is anatomically
        plausible as a trunk/root rather than purely geometric.
        """
        nodes = list(nodes)
        sub = self.label_graph.subgraph(nodes)

        if self.point_radii is not None:
            first = max(nodes, key=lambda n: self.point_radii[n])
        else:
            first = nodes[0]

        seeds = [first]
        dist_to_seeds = dict(nx.single_source_dijkstra_path_length(
            sub, first, weight='length'))

        while len(seeds) < n_seeds:
            candidates = [(d, n) for n, d in dist_to_seeds.items()
                         if n not in seeds]
            if not candidates:
                break
            candidates.sort(reverse=True)
            new_seed = candidates[0][1]
            seeds.append(new_seed)

            new_dist = nx.single_source_dijkstra_path_length(
                sub, new_seed, weight='length')
            for n, d in new_dist.items():
                if d < dist_to_seeds.get(n, np.inf):
                    dist_to_seeds[n] = d

        return seeds

    def _voronoi_partition(self, nodes, seeds):
        """
        Assign every node in `nodes` to whichever seed is graph-closest
        (multi-source Dijkstra). Each resulting partition is connected: any
        node on the shortest path from a seed to one of its cell's members is,
        by construction, at least as close to that seed as the member is, so it
        belongs to the same cell.
        """
        sub = self.label_graph.subgraph(nodes)

        best_dist = {n: np.inf for n in nodes}
        best_seed = {}
        for seed in seeds:
            for n, d in nx.single_source_dijkstra_path_length(
                    sub, seed, weight='length').items():
                if d < best_dist[n]:
                    best_dist[n] = d
                    best_seed[n] = seed

        partitions = {seed: [] for seed in seeds}
        for n, seed in best_seed.items():
            partitions[seed].append(n)

        return [p for p in partitions.values() if p]

    # -- Stage 2, default: split at the component's own real branch points --

    def _split_by_bifurcation(self, nodes, n_needed, verbose=True):
        """
        Split one connected component into up to `n_needed` groups by cutting
        it at real branch points instead of an arbitrary geometric partition.
        See the class docstring's Stage 2 section for the rationale.

        `nodes` is already a tree in `self.label_graph` (stage 1 reduces every
        component to an MST). Two passes:

        1. Root the tree at the highest-radius point (falls back to an
           arbitrary node if `point_radii` is unavailable), then find every
           node of degree >= 3 in the *undirected* tree -- a real fork, not a
           chosen seed. For each such fork, each child branch (in the rooted
           orientation) is a split candidate, scored by
           `sum(radius**3 for point in branch)` -- a Murray's-law-consistent
           size proxy, so the largest/most significant branches get cut off
           first. Candidates are accepted greedily, largest first, skipping
           any that overlap an already-accepted one (this also transparently
           rejects candidates nested inside an accepted branch, since nested
           descendants are a subset of covered nodes).
        2. If fewer than `n_needed - 1` real forks are usable (too few, or
           each branch too small), the remaining trunk is repeatedly cut at
           the thinnest edge along its own diameter path (two-BFS tree
           diameter, then the minimum-radius edge along that path) -- a
           physical narrowing, not a distance midpoint.

        If neither pass reaches `n_needed` groups (the component genuinely
        has no more real forks or narrowings above `min_tree_size`), fewer
        groups are returned rather than forcing a boundary that isn't there
        -- callers must handle receiving fewer than `n_needed` groups.

        Returns a list of node-index lists (a partition of `nodes`, in the
        sense that every node ends up in exactly one group).
        """
        nodes = list(nodes)
        if n_needed <= 1 or len(nodes) < 2:
            return [nodes]

        sub = self.label_graph.subgraph(nodes)
        root = (max(nodes, key=lambda n: self.point_radii[n])
                if self.point_radii is not None else nodes[0])
        directed = nx.bfs_tree(sub, root)

        def branch_size(desc):
            if self.point_radii is not None:
                return float(sum(self.point_radii[m] ** 3 for m in desc))
            return float(len(desc))

        candidates = []
        for f in nodes:
            if sub.degree(f) < 3:
                continue
            for c in directed.successors(f):
                desc = nx.descendants(directed, c) | {c}
                if self.min_tree_size <= len(desc) <= 0.9 * len(nodes):
                    candidates.append((branch_size(desc), desc))
        candidates.sort(key=lambda x: -x[0])

        chosen, covered = [], set()
        for _, desc in candidates:
            if len(chosen) >= n_needed - 1:
                break
            if desc.isdisjoint(covered):
                chosen.append(sorted(desc))
                covered |= desc

        n_forks_used = len(chosen)

        # Fallback: cut the remaining trunk at its thinnest point, repeatedly,
        # for whatever shortfall the real forks above didn't cover.
        trunk = [n for n in nodes if n not in covered]
        n_bottleneck_cuts = 0
        while len(chosen) < n_needed - 1 and len(trunk) >= 2 * self.min_tree_size:
            trunk_sub = self.label_graph.subgraph(trunk)
            a = trunk[0]
            dist1 = dict(nx.single_source_shortest_path_length(trunk_sub, a))
            far1 = max(dist1, key=dist1.get)
            dist2 = dict(nx.single_source_shortest_path_length(trunk_sub, far1))
            far2 = max(dist2, key=dist2.get)
            path = nx.shortest_path(trunk_sub, far1, far2)
            if len(path) < 2:
                break

            if self.point_radii is not None:
                thinness = [min(self.point_radii[path[i]], self.point_radii[path[i + 1]])
                           for i in range(len(path) - 1)]
            else:
                thinness = [0.0] * (len(path) - 1)
            cut_i = int(np.argmin(thinness))
            u, v = path[cut_i], path[cut_i + 1]

            trunk_cut = trunk_sub.copy()
            trunk_cut.remove_edge(u, v)
            parts = sorted(nx.connected_components(trunk_cut), key=len)
            if len(parts) != 2 or len(parts[0]) < self.min_tree_size:
                break  # no useful cut left; stop rather than force one

            chosen.append(sorted(parts[0]))
            covered |= parts[0]
            trunk = sorted(parts[1])
            n_bottleneck_cuts += 1

        chosen.append(sorted(trunk))

        if verbose:
            shortfall = n_needed - len(chosen)
            print(f"  Bifurcation split: {n_forks_used} real-fork cut(s), "
                  f"{n_bottleneck_cuts} bottleneck fallback cut(s) -> "
                  f"{len(chosen)} group(s)"
                  + (f"  (wanted {n_needed}, {shortfall} short -- no more "
                     f"honest cuts available, not forcing one)" if shortfall > 0 else ""))

        return chosen

    # -- Per-tree reconstruction (reuses the existing pipeline verbatim) -----

    def _reconstruct_group(self, node_indices, method, verbose=True):
        """
        Turn one group of point indices (a natural component or a Voronoi
        partition) into a tree, via `ImprovedVascularTreeReconstruction` --
        identical math to the rest of this module. Returns a dict in the same
        schema `MultiTreeReconstruction` uses, so the inherited `visualize_all_trees`/`export_all_trees`/
        `get_tree_statistics` keep working unmodified.
        """
        node_indices = sorted(node_indices)
        if len(node_indices) < self.min_tree_size:
            return None

        sub_points = self.original_points[node_indices]
        sub_radii = (self.point_radii[node_indices]
                    if self.point_radii is not None else None)

        reconstructor = ImprovedVascularTreeReconstruction(
            sub_points,
            gamma=self.gamma, mu=self.mu, Q_perf=self.Q_perf,
            P_out=self.P_out, P_in=self.P_in,
            point_radii=sub_radii,
            **self.builder_kwargs
        )

        try:
            tree, root, radii, metrics = reconstructor.reconstruct(
                k_neighbors=10, n_candidates=5, optimize_bifurcations=True,
                method=method,
            )
        except Exception as e:
            if verbose:
                print(f"  Failed to reconstruct group of {len(node_indices)} "
                      f"points: {e}")
            return None

        # Only mark as consumed what was actually reconstructed (lab_diary 4.4/4.5).
        reconstructed_local = sorted(
            n for n in tree.nodes() if n < len(sub_points))
        coverage = len(reconstructed_local) / max(len(sub_points), 1)

        return {
            'tree': tree,
            'root': root,
            'radii': radii,
            'metrics': metrics,
            'points': reconstructor.points,
            'source_points': sub_points[reconstructed_local],
            'point_indices': [node_indices[i] for i in reconstructed_local],
            'coverage': coverage,
            'reconstructor': reconstructor,
        }

    # -- Driver ----------------------------------------------------------

    def reconstruct_multiple_trees(self, methods=None, verbose=True):
        """
        Run both stages and populate `self.trees` / `self.remaining_points`.

        Parameters:
        -----------
        methods : list of str or None
            Root-selection strategy per tree (cycled if shorter than the
            number of trees produced), passed through to each
            `ImprovedVascularTreeReconstruction.reconstruct()` call. Defaults
            to `["thickest_z", "thickest", "thickest", ...]`.
        """
        if methods is None:
            methods = ["thickest_z"] + ["thickest"] * max(self.n_trees - 1, 0)

        components = self._build_label_graph()
        usable = [c for c in components if len(c) >= self.min_tree_size]

        if verbose:
            print(f"Label-continuity graph: {len(components)} connected "
                  f"component(s); sizes (top 10): "
                  f"{self.component_sizes[:10]}"
                  f"{'...' if len(self.component_sizes) > 10 else ''}")
            print(f"{len(usable)} component(s) >= min_tree_size="
                  f"{self.min_tree_size}\n")

        if not usable and components:
            # Nothing meets the threshold on its own; fall back to whatever
            # exists so stage 2 has something to split rather than producing
            # nothing at all.
            usable = components[:1]

        self.trees = []
        assigned = set()

        if len(usable) >= self.n_trees:
            # Stage 1 alone already provides enough continuous pieces.
            if verbose:
                print(f"Stage 1: {len(usable)} natural component(s) already "
                      f">= n_trees={self.n_trees}; using the "
                      f"{self.n_trees} largest directly, no split needed.")
            for group in usable[:self.n_trees]:
                method = methods[len(self.trees) % len(methods)]
                data = self._reconstruct_group(group, method, verbose=verbose)
                if data is not None:
                    self.trees.append(data)
                    assigned.update(data['point_indices'])
        else:
            if verbose and len(usable) > 1:
                print(f"Stage 1: {len(usable)} natural component(s) found; "
                      f"using the {len(usable) - 1} smaller one(s) as their "
                      f"own tree(s) and splitting the largest.")
            # Every natural component except the largest becomes its own tree.
            for group in usable[1:]:
                if len(self.trees) >= self.n_trees:
                    break
                method = methods[len(self.trees) % len(methods)]
                data = self._reconstruct_group(group, method, verbose=verbose)
                if data is not None:
                    self.trees.append(data)
                    assigned.update(data['point_indices'])

            # Stage 2: split the single largest component to make up the rest.
            n_needed = self.n_trees - len(self.trees)
            if usable and n_needed > 0:
                largest = usable[0]
                if verbose:
                    print(f"\nStage 2: splitting the largest component "
                          f"({len(largest)} points) into up to {n_needed} "
                          f"subtree(s) via {self.split_method!r} split to "
                          f"reach n_trees={self.n_trees}.")
                if self.split_method == "bifurcation":
                    partitions = self._split_by_bifurcation(largest, n_needed, verbose=verbose)
                else:
                    seeds = self._select_split_seeds(largest, n_needed)
                    partitions = self._voronoi_partition(largest, seeds)
                if verbose:
                    print(f"  Partition sizes: {[len(p) for p in partitions]}")
                for part in partitions:
                    if len(self.trees) >= self.n_trees:
                        break
                    method = methods[len(self.trees) % len(methods)]
                    data = self._reconstruct_group(part, method, verbose=verbose)
                    if data is not None:
                        self.trees.append(data)
                        assigned.update(data['point_indices'])

        leftover = sorted(set(range(len(self.original_points))) - assigned)
        self.remaining_points = self.original_points[leftover]

        if verbose:
            print(f"\nFINAL: {len(self.trees)} tree(s), {len(assigned)} "
                  f"point(s) assigned, {len(self.remaining_points)} left over "
                  f"(of {len(self.original_points)} input points)")

        return self.trees


# ---------------------------------------------------------------------------
# Adaptive n_trees search (additive -- nothing above this line is modified)
#
# Both MultiTreeReconstruction and LabelConnectedMultiTreeReconstruction can come
# back with fewer trees than requested when their default parameters are too
# conservative for a given dataset (e.g. MultiTreeReconstruction.min_tree_size=50
# discarding a real but small fragment; LabelConnectedMultiTreeReconstruction's
# bifurcation split declining to force a cut that isn't backed by a real fork).
# Rather than silently accepting that shortfall, these wrappers retry with
# progressively relaxed parameters -- logging every attempt -- and only give up
# once the relaxation ladder is exhausted. They never fabricate a split that
# isn't there; each rung is a real, already-existing knob in the underlying
# class, just used more aggressively.
# ---------------------------------------------------------------------------

def adaptive_reconstruct_basic(skeleton_points, n_trees=3, min_tree_size=50,
                                k_neighbors_initial=5, min_tree_size_floor=10,
                                k_neighbors_floor=2, max_attempts=4, verbose=True):
    """
    Retry MultiTreeReconstruction with progressively relaxed parameters until
    n_trees is reached or the relaxation ladder is exhausted.

    Ladder (cheapest/least-invasive first):
      1. default min_tree_size / k_neighbors_initial
      2. halve min_tree_size toward min_tree_size_floor (stop discarding real
         small fragments as noise)
      3. decrement k_neighbors_initial toward k_neighbors_floor (a sparser
         initial graph splits more readily into separate components)

    Returns
    -------
    rec : MultiTreeReconstruction
        The reconstructor from the best (highest tree count, ties broken by
        earliest/least-relaxed) attempt.
    log : list of dict
        One entry per attempt: params tried, trees found, leftover points.
    """
    mts, k = min_tree_size, k_neighbors_initial
    log = []
    best_rec, best_n = None, -1

    for attempt in range(1, max_attempts + 1):
        rec = MultiTreeReconstruction(skeleton_points, n_trees=n_trees)
        trees = rec.reconstruct_multiple_trees(
            k_neighbors_initial=k, min_tree_size=mts)
        n_found = len(trees)
        entry = {"attempt": attempt, "k_neighbors_initial": k, "min_tree_size": mts,
                 "trees_found": n_found, "leftover_points": len(rec.remaining_points)}
        log.append(entry)
        if verbose:
            print(f"[adaptive basic] attempt {attempt}: k_neighbors_initial={k}, "
                  f"min_tree_size={mts} -> {n_found}/{n_trees} trees, "
                  f"{len(rec.remaining_points)} leftover")

        if n_found > best_n:
            best_rec, best_n = rec, n_found
        if n_found >= n_trees:
            return rec, log

        if mts > min_tree_size_floor:
            mts = max(min_tree_size_floor, mts // 2)
        elif k > k_neighbors_floor:
            k -= 1
        else:
            if verbose:
                print(f"[adaptive basic] relaxation ladder exhausted at "
                      f"attempt {attempt}; best result kept ({best_n}/{n_trees} trees)")
            break

    return best_rec, log


def adaptive_reconstruct_labelconnected(skeleton_points, mask=None, n_trees=3,
                                         point_radii=None, min_tree_size=10,
                                         min_tree_size_floor=3, max_attempts=3,
                                         builder_kwargs=None, verbose=True):
    """
    Retry LabelConnectedMultiTreeReconstruction with progressively relaxed
    parameters until n_trees is reached or the relaxation ladder is exhausted.

    Ladder:
      1. split_method="bifurcation" (default -- only cuts at real forks; may
         deliberately return fewer than n_trees if the geometry doesn't
         support enough real cuts)
      2. split_method="voronoi" (always able to divide a component further,
         at the cost of the cut being geometric rather than anatomical)
      3. halve min_tree_size toward min_tree_size_floor (stop discarding real
         small components/partitions as noise)

    Returns
    -------
    rec : LabelConnectedMultiTreeReconstruction
        The reconstructor from the best (highest tree count) attempt.
    log : list of dict
        One entry per attempt: params tried, trees found, tree sizes.
    """
    mts = min_tree_size
    split_methods = ["bifurcation", "voronoi"]
    log = []
    best_rec, best_n = None, -1

    for attempt in range(1, max_attempts + 1):
        split_method = split_methods[min(attempt - 1, len(split_methods) - 1)]
        rec = LabelConnectedMultiTreeReconstruction(
            skeleton_points, mask=mask, n_trees=n_trees, point_radii=point_radii,
            min_tree_size=mts, split_method=split_method,
            builder_kwargs=builder_kwargs)
        trees = rec.reconstruct_multiple_trees(verbose=False)
        n_found = len(trees)
        sizes = [len(t["point_indices"]) for t in trees]
        entry = {"attempt": attempt, "split_method": split_method, "min_tree_size": mts,
                 "trees_found": n_found, "tree_sizes": sizes}
        log.append(entry)
        if verbose:
            print(f"[adaptive label-connected] attempt {attempt}: "
                  f"split_method={split_method!r}, min_tree_size={mts} -> "
                  f"{n_found}/{n_trees} trees, sizes={sizes}")

        if n_found > best_n:
            best_rec, best_n = rec, n_found
        if n_found >= n_trees:
            return rec, log

        if attempt >= len(split_methods) and mts > min_tree_size_floor:
            mts = max(min_tree_size_floor, mts // 2)

    if verbose and best_n < n_trees:
        print(f"[adaptive label-connected] relaxation ladder exhausted; "
              f"best result kept ({best_n}/{n_trees} trees)")

    return best_rec, log
