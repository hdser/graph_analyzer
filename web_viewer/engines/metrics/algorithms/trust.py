"""
Trust Metric Algorithms

Trust network algorithms including EigenTrust and Appleseed.
"""

from typing import Dict, Any, List, Optional
import logging

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class EigenTrustAlgorithm(BaseMetricAlgorithm):
    """
    EigenTrust algorithm for computing social trust scores.
    
    Based on: Kamvar et al., "The EigenTrust Algorithm for Reputation
    Management in P2P Networks", 2003.
    
    The algorithm computes global trust values by iteratively updating
    trust scores based on the trust matrix and a pre-trust vector.
    """
    
    name = "eigentrust"
    category = "trust"
    description = "EigenTrust score for social trust networks"
    cost = "medium"
    
    def __init__(self, alpha: float = 0.15, max_iterations: int = 100, tolerance: float = 1e-9, **kwargs):
        """
        Initialize EigenTrust algorithm.
        
        Args:
            alpha: Teleportation probability (weight for pre-trust vector)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def compute(
        self,
        G: nx.DiGraph,
        U: nx.Graph,
        nodes: list,
        converters: List[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute EigenTrust scores.
        
        Args:
            G: Directed graph
            U: Undirected version
            nodes: List of nodes to compute for
            converters: List of trusted seed nodes (high pre-trust)
        """
        n = G.number_of_nodes()
        
        if n == 0:
            return {node: {} for node in nodes}
        
        logger.debug(f"Computing EigenTrust for {n} nodes")
        
        # Build node-to-index mapping
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        # Build adjacency matrix (trust matrix)
        trust_matrix = nx.to_scipy_sparse_array(G, nodelist=node_list, format='csr')
        
        # Remove self-loops
        W_no_self = self._remove_self_loops(trust_matrix)
        
        # Build column-stochastic matrix
        C = self._build_column_stochastic_matrix(W_no_self)
        
        # Initialize pre-trust vector
        pre_trust = self._build_pre_trust_vector(n, converters, node_to_idx)
        
        # Power iteration
        trust = pre_trust.copy()
        iterations_used = 0
        
        for iteration in range(self.max_iterations):
            trust_prev = trust.copy()
            
            # EigenTrust update: t = (1-α)C^T t + α p
            trust = (1 - self.alpha) * (C.T @ trust_prev) + self.alpha * pre_trust
            
            # Check convergence
            if np.linalg.norm(trust - trust_prev, ord=1) < self.tolerance:
                iterations_used = iteration + 1
                logger.debug(f"EigenTrust converged at iteration {iteration + 1}")
                break
        else:
            iterations_used = self.max_iterations
            logger.warning(f"EigenTrust did not converge in {self.max_iterations} iterations")
        
        # Build result
        result = {}
        for node in nodes:
            if node in node_to_idx:
                idx = node_to_idx[node]
                result[node] = {"eigentrust": float(trust[idx])}
            else:
                result[node] = {"eigentrust": 0.0}
        
        return result
    
    def _remove_self_loops(self, trust_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Remove self-loops efficiently."""
        coo = trust_matrix.tocoo()
        mask = coo.row != coo.col
        return sp.csr_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])),
            shape=trust_matrix.shape
        )
    
    def _build_column_stochastic_matrix(self, trust_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Build column-stochastic matrix from trust matrix."""
        col_sums = np.asarray(trust_matrix.sum(axis=0)).ravel()
        
        # Handle zero columns (dangling nodes)
        zero_cols = (col_sums == 0)
        col_sums[zero_cols] = 1.0
        
        D_inv = sp.diags(1.0 / col_sums, format='csr')
        C = trust_matrix @ D_inv
        
        # For zero columns, distribute trust equally
        if np.any(zero_cols):
            n = trust_matrix.shape[0]
            C = C.tolil()
            for j in np.where(zero_cols)[0]:
                C[:, j] = 1.0 / n
            C = C.tocsr()
        
        return C
    
    def _build_pre_trust_vector(
        self,
        n: int,
        converters: List[str] = None,
        node_to_idx: Dict[str, int] = None
    ) -> np.ndarray:
        """Build pre-trust vector for EigenTrust."""
        pre_trust = np.ones(n) / n
        
        if converters and node_to_idx:
            converter_indices = [node_to_idx[c] for c in converters if c in node_to_idx]
            if converter_indices:
                pre_trust = np.ones(n) * 0.1
                pre_trust[converter_indices] = 0.9 / len(converter_indices)
                pre_trust = pre_trust / pre_trust.sum()
        
        return pre_trust


class AppleseedAlgorithm(BaseMetricAlgorithm):
    """
    Appleseed algorithm for computing social trust scores.
    
    Based on: Ziegler and Lausen, "Spreading Activation Models for
    Trust Propagation", 2005.
    
    The algorithm propagates trust energy from seed nodes through
    the network, with energy decaying at each step.
    """
    
    name = "appleseed"
    category = "trust"
    description = "Appleseed trust propagation with energy decay"
    cost = "medium"
    
    def __init__(self, energy_decay: float = 0.85, max_iterations: int = 200, tolerance: float = 1e-9, **kwargs):
        """
        Initialize Appleseed algorithm.
        
        Args:
            energy_decay: Energy retention factor (d in the paper)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        super().__init__(**kwargs)
        self.energy_decay = energy_decay
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def compute(
        self,
        G: nx.DiGraph,
        U: nx.Graph,
        nodes: list,
        converters: List[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute Appleseed trust scores.
        
        Args:
            G: Directed graph
            U: Undirected version
            nodes: List of nodes to compute for
            converters: List of seed nodes to start energy propagation
        """
        n = G.number_of_nodes()
        
        if n == 0:
            return {node: {} for node in nodes}
        
        logger.debug(f"Computing Appleseed for {n} nodes")
        
        # Build node-to-index mapping
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        # Build trust matrix
        trust_matrix = nx.to_scipy_sparse_array(G, nodelist=node_list, format='csr')
        
        # Build row-stochastic transition matrix
        P = self._build_row_stochastic_matrix(trust_matrix)
        
        # Initialize seed energy
        energy = self._initialize_energy(n, converters, node_to_idx)
        
        # Trust accumulator
        trust = np.zeros(n, dtype=np.float64)
        iterations_used = 0
        
        # Energy propagation
        for iteration in range(self.max_iterations):
            # Accumulate trust from current energy
            trust += (1 - self.energy_decay) * energy
            
            # Propagate energy
            new_energy = self.energy_decay * (P.T @ energy)
            
            # Check convergence
            if np.linalg.norm(new_energy - energy, ord=1) < self.tolerance:
                iterations_used = iteration + 1
                logger.debug(f"Appleseed converged at iteration {iteration + 1}")
                trust += new_energy
                break
            
            energy = new_energy
        else:
            iterations_used = self.max_iterations
            logger.warning(f"Appleseed did not converge in {self.max_iterations} iterations")
            trust += energy
        
        # Normalize
        total = trust.sum()
        if total > 0:
            trust = trust / total
        
        # Build result
        result = {}
        for node in nodes:
            if node in node_to_idx:
                idx = node_to_idx[node]
                result[node] = {"appleseed": float(trust[idx])}
            else:
                result[node] = {"appleseed": 0.0}
        
        return result
    
    def _build_row_stochastic_matrix(self, trust_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Build row-stochastic matrix from trust matrix."""
        row_sums = np.asarray(trust_matrix.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        D_inv = sp.diags(1.0 / row_sums, format='csr')
        return D_inv @ trust_matrix
    
    def _initialize_energy(
        self,
        n: int,
        converters: List[str] = None,
        node_to_idx: Dict[str, int] = None
    ) -> np.ndarray:
        """Initialize energy distribution for Appleseed."""
        energy = np.zeros(n, dtype=np.float64)
        
        if converters and node_to_idx:
            converter_indices = [node_to_idx[c] for c in converters if c in node_to_idx]
            if converter_indices:
                energy[converter_indices] = 1.0 / len(converter_indices)
            else:
                energy[:] = 1.0 / n
        else:
            energy[:] = 1.0 / n
        
        return energy