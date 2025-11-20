"""
Graph Metrics Calculator - Comprehensive NetworkX Implementation
Computes 120+ graph metrics for directed trust networks
WITH PARALLEL PROCESSING for faster computation
WITH SELECTIVE METRIC COMPUTATION for faster targeted analysis
"""

import pandas as pd
import numpy as np
import networkx as nx
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import logging
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing
import sys

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# METRIC CATEGORIES CONFIGURATION
# ==============================================================================

METRIC_CATEGORIES = {
    'topology': 'Basic Topology (in/out degree, degree imbalance)',
    'centrality': 'Centrality Measures (pagerank, betweenness, eigenvector, etc.)',
    'clustering': 'Clustering Metrics (clustering coefficient, triangles, squares)',
    'community': 'Community Detection (Louvain, core numbers, onion layers)',
    'paths': 'Path Metrics (shortest paths, reachability, eccentricity)',
    'distances': 'Distance Measures (radius, diameter, center, periphery)',
    'structural': 'Structural Metrics (structural holes, articulation points, neighbor stats)',
    'reciprocity': 'Reciprocity Metrics (mutual connections, one-way connections)',
    'reach': 'Reach Metrics (n-hop reach, network penetration)',
    'components': 'Component Metrics (weak/strong components)',
    'vitality': 'Vitality Metrics (closeness vitality)',
    'dispersion': 'Dispersion Metrics (node dispersion)',
    'efficiency': 'Efficiency Metrics (local efficiency)',
    'flow': 'Flow Metrics (flow hierarchy)',
    'dominance': 'Dominance Metrics (dominated nodes, dominance ratio)'
}

# Quick preset groups
METRIC_PRESETS = {
    'basic': ['topology', 'clustering'],
    'essential': ['topology', 'centrality', 'clustering', 'community'],
    'moderate': ['topology', 'centrality', 'clustering', 'community', 'paths', 'structural'],
    'all': list(METRIC_CATEGORIES.keys())
}


# ==============================================================================
# PARALLEL HELPER FUNCTIONS (must be at module level for pickling)
# ==============================================================================

def _compute_node_paths(G, node):
    """Compute path metrics for a single node"""
    result = {}
    try:
        lengths = dict(nx.single_source_shortest_path_length(G, node))
        if len(lengths) > 1:
            vals = [l for l in lengths.values() if l > 0]
            if vals:
                result['avg_shortest_path'] = np.mean(vals)
                result['median_shortest_path'] = np.median(vals)
                result['max_shortest_path'] = np.max(vals)
                result['path_variance'] = np.var(vals)
                result['path_sum'] = sum(vals)
                result['reachable_nodes'] = len(vals)
            else:
                result.update({k: 0 for k in [
                    'avg_shortest_path', 'median_shortest_path',
                    'max_shortest_path', 'path_variance',
                    'path_sum', 'reachable_nodes'
                ]})

        # Direct paths
        result['paths_length_1'] = len(list(G.successors(node)))

        # 2-hop paths
        paths_2 = set()
        for neighbor in G.successors(node):
            for second_hop in G.successors(neighbor):
                if second_hop != node:
                    paths_2.add(second_hop)
        result['paths_length_2_targets'] = len(paths_2)

    except Exception:
        result.update({k: 0 for k in [
            'avg_shortest_path', 'median_shortest_path',
            'max_shortest_path', 'path_variance', 'path_sum',
            'reachable_nodes', 'paths_length_1', 'paths_length_2_targets'
        ]})

    return node, result


def _compute_node_reach(G, node, max_hops):
    """Compute BFS reach for a single node"""
    result = {}
    visited = {node}
    current = {node}
    total = 0
    n = G.number_of_nodes()

    for hop in range(1, max_hops + 1):
        next_level = set()
        for parent in current:
            neighbors = set(G.successors(parent))
            next_level.update(neighbors - visited)

        if next_level:
            visited.update(next_level)
            result[f'reach_hop_{hop}'] = len(next_level)
            total += len(next_level)
        else:
            result[f'reach_hop_{hop}'] = 0

        current = next_level

    result['total_reach'] = total
    result['network_penetration'] = total / (n - 1) if n > 1 else 0

    return node, result


def _compute_node_wiener(U, node):
    """Compute Wiener contribution for a single node"""
    try:
        lengths = nx.single_source_shortest_path_length(U, node)
        return node, sum(lengths.values())
    except Exception:
        return node, 0


def _compute_node_dominance(G, node, n):
    """Compute dominance for a single node"""
    try:
        descendants = nx.descendants(G, node)
        dominated = len(descendants)
        return node, {
            'dominated_nodes_count': dominated,
            'dominance_ratio': dominated / (n - 1) if n > 1 else 0
        }
    except Exception:
        return node, {'dominated_nodes_count': 0, 'dominance_ratio': 0}


def _compute_node_lrc(G, node):
    """Compute local reaching centrality for a single node"""
    try:
        return node, nx.local_reaching_centrality(G, node)
    except Exception:
        return node, 0


# ==============================================================================
# MAIN METRICS CLASS
# ==============================================================================

class GraphMetrics:
    """Compute comprehensive graph metrics with parallel processing and selective computation"""

    def __init__(self, graph: nx.DiGraph, n_jobs: int = None, metrics_mode: str = 'all'):
        self.G = graph
        self.U = graph.to_undirected()
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()
        self.n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)

        # Parse metrics mode
        self.metrics_to_compute = self._parse_metrics_mode(metrics_mode)

        logger.info("=" * 70)
        logger.info("GRAPH METRICS CALCULATOR INITIALIZED")
        logger.info("=" * 70)
        logger.info("Graph Statistics:")
        logger.info(f"  • Nodes: {self.n:,}")
        logger.info(f"  • Edges: {self.m:,}")
        logger.info(f"  • Avg degree: {2 * self.m / self.n:.2f}" if self.n > 0 else "  • Avg degree: N/A")
        logger.info(f"  • Density: {self.m / (self.n * (self.n - 1)):.6f}" if self.n > 1 else "  • Density: N/A")
        logger.info("  • Is directed: True")
        logger.info("Parallel Processing:")
        logger.info(f"  • CPU cores available: {multiprocessing.cpu_count()}")
        logger.info(f"  • Workers to use: {self.n_jobs}")
        logger.info("Metrics Configuration:")
        logger.info(f"  • Mode: {metrics_mode}")
        logger.info(f"  • Categories to compute: {len(self.metrics_to_compute)}/{len(METRIC_CATEGORIES)}")
        for cat in self.metrics_to_compute:
            logger.info(f"    ✓ {cat}: {METRIC_CATEGORIES[cat]}")
        logger.info("=" * 70)

    def _parse_metrics_mode(self, mode: str) -> list:
        """Parse metrics mode configuration"""
        mode = mode.lower().strip()

        # Preset
        if mode in METRIC_PRESETS:
            return METRIC_PRESETS[mode]

        # Comma-separated categories
        if ',' in mode:
            categories = [cat.strip() for cat in mode.split(',')]
            valid_categories = []
            for cat in categories:
                if cat in METRIC_CATEGORIES:
                    valid_categories.append(cat)
                else:
                    logger.warning(f"Unknown metric category: {cat}")
            return valid_categories if valid_categories else METRIC_PRESETS['all']

        # Single category
        if mode in METRIC_CATEGORIES:
            return [mode]

        # Default to all
        logger.warning(f"Unknown metrics mode: {mode}. Using 'all'")
        return METRIC_PRESETS['all']

    def compute_all(self) -> pd.DataFrame:
        """Compute selected metrics with progress logging"""
        start_time = time.time()
        metrics = {node: {} for node in self.G.nodes()}

        logger.info("")
        logger.info("STARTING METRIC COMPUTATION")
        logger.info("=" * 70)

        step_num = 1
        total_steps = len(self.metrics_to_compute)

        # 1. Basic topology
        if 'topology' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing topology metrics...")
            self._topology(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 2. Centrality
        if 'centrality' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing centrality metrics...")
            self._centrality(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 3. Clustering
        if 'clustering' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing clustering metrics...")
            self._clustering(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 4. Community
        if 'community' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing community metrics...")
            self._community(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 5. Paths (PARALLEL)
        if 'paths' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing path metrics (PARALLEL)...")
            self._paths_parallel(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 6. Distances
        if 'distances' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing distance measures...")
            self._distances(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 7. Structural
        if 'structural' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing structural metrics...")
            self._structural(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 8. Reciprocity
        if 'reciprocity' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing reciprocity metrics...")
            self._reciprocity(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 9. Reach (PARALLEL)
        if 'reach' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing reach metrics (PARALLEL)...")
            self._reach_parallel(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 10. Components
        if 'components' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing component metrics...")
            self._components(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 11. Vitality
        if 'vitality' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing vitality metrics...")
            self._vitality(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 12. Dispersion
        if 'dispersion' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing dispersion metrics...")
            self._dispersion(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 13. Efficiency
        if 'efficiency' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing efficiency metrics...")
            self._efficiency(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 14. Flow
        if 'flow' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing flow metrics...")
            self._flow(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # 15. Dominance (PARALLEL)
        if 'dominance' in self.metrics_to_compute:
            step_start = time.time()
            logger.info(f"[{step_num:2d}/{total_steps:2d}] Computing dominance metrics (PARALLEL)...")
            self._dominance_parallel(metrics)
            logger.info(f"         ✓ Completed in {time.time() - step_start:.2f}s")
            step_num += 1

        # Convert to DataFrame
        logger.info("")
        logger.info("Converting to DataFrame...")
        df = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
        df.rename(columns={'index': 'avatar'}, inplace=True)

        # Clean data
        logger.info("Cleaning data (converting booleans, filling NaN/inf)...")
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        df = df.replace([np.inf, -np.inf], 0).fillna(0)

        total_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info("COMPUTATION COMPLETE!")
        logger.info(f"  • Total time: {total_time:.2f}s ({total_time / 60:.1f} minutes)")
        logger.info(f"  • Nodes processed: {len(df):,}")
        logger.info(f"  • Metrics computed: {len(df.columns) - 1}")
        logger.info(f"  • Categories computed: {len(self.metrics_to_compute)}/{len(METRIC_CATEGORIES)}")
        logger.info(f"  • Avg time per node: {total_time / len(df):.3f}s")
        logger.info("=" * 70)

        return df

    # ------------------------------------------------------------------
    # Individual metric category implementations
    # ------------------------------------------------------------------

    def _topology(self, metrics):
        """Basic topology metrics"""
        for node in self.G.nodes():
            in_d = self.G.in_degree(node)
            out_d = self.G.out_degree(node)
            metrics[node]['in_degree'] = in_d
            metrics[node]['out_degree'] = out_d
            metrics[node]['total_degree'] = in_d + out_d
            metrics[node]['degree_imbalance'] = abs(in_d - out_d) / (in_d + out_d) if (in_d + out_d) > 0 else 0
        logger.debug(f"  Computed topology for {len(metrics)} nodes")

    def _centrality(self, metrics):
        """All centrality measures"""
        computed = []

        try:
            logger.debug("  Computing degree centrality...")
            in_deg = nx.in_degree_centrality(self.G)
            out_deg = nx.out_degree_centrality(self.G)
            deg_undirected = nx.degree_centrality(self.U)
            for node in self.G.nodes():
                metrics[node]['in_degree_centrality'] = in_deg[node]
                metrics[node]['out_degree_centrality'] = out_deg[node]
                metrics[node]['degree_centrality_undirected'] = deg_undirected[node]
            computed.append("degree")
        except Exception as e:
            logger.warning(f"  ⚠ Degree centrality failed: {e}")

        try:
            logger.debug("  Computing closeness centrality...")
            closeness = nx.closeness_centrality(self.G)
            closeness_in = nx.closeness_centrality(self.G.reverse())
            closeness_undirected = nx.closeness_centrality(self.U)
            for node in self.G.nodes():
                metrics[node]['closeness_centrality'] = closeness[node]
                metrics[node]['closeness_centrality_in'] = closeness_in[node]
                metrics[node]['closeness_centrality_undirected'] = closeness_undirected[node]
            computed.append("closeness")
        except Exception as e:
            logger.warning(f"  ⚠ Closeness centrality failed: {e}")

        try:
            logger.debug("  Computing betweenness centrality...")
            betweenness = nx.betweenness_centrality(self.G)
            betweenness_undirected = nx.betweenness_centrality(self.U)
            for node in self.G.nodes():
                metrics[node]['betweenness_centrality'] = betweenness[node]
                metrics[node]['betweenness_centrality_undirected'] = betweenness_undirected[node]
            computed.append("betweenness")
        except Exception as e:
            logger.warning(f"  ⚠ Betweenness centrality failed: {e}")

        try:
            logger.debug("  Computing eigenvector centrality...")
            eigenvector = nx.eigenvector_centrality(self.G, max_iter=1000)
            eigenvector_undirected = nx.eigenvector_centrality(self.U, max_iter=1000)
            for node in self.G.nodes():
                metrics[node]['eigenvector_centrality'] = eigenvector[node]
                metrics[node]['eigenvector_centrality_undirected'] = eigenvector_undirected[node]
            computed.append("eigenvector")
        except Exception as e:
            logger.warning(f"  ⚠ Eigenvector centrality failed: {e}")

        try:
            logger.debug("  Computing Katz centrality...")
            alpha = self._safe_alpha()
            katz = nx.katz_centrality(self.G, alpha=alpha, max_iter=1000)
            katz_undirected = nx.katz_centrality(self.U, alpha=alpha, max_iter=1000)
            for node in self.G.nodes():
                metrics[node]['katz_centrality'] = katz[node]
                metrics[node]['katz_centrality_undirected'] = katz_undirected[node]
            computed.append("katz")
        except Exception as e:
            logger.warning(f"  ⚠ Katz centrality failed: {e}")

        try:
            logger.debug("  Computing PageRank...")
            pagerank = nx.pagerank(self.G, alpha=0.85)
            pagerank_undirected = nx.pagerank(self.U, alpha=0.85)
            for node in self.G.nodes():
                metrics[node]['pagerank'] = pagerank[node]
                metrics[node]['pagerank_undirected'] = pagerank_undirected[node]
            computed.append("pagerank")
        except Exception as e:
            logger.warning(f"  ⚠ PageRank failed: {e}")

        try:
            logger.debug("  Computing HITS...")
            hubs, authorities = nx.hits(self.G, max_iter=100)
            for node in self.G.nodes():
                metrics[node]['hub_score'] = hubs[node]
                metrics[node]['authority_score'] = authorities[node]
            computed.append("hits")
        except Exception as e:
            logger.warning(f"  ⚠ HITS failed: {e}")

        try:
            logger.debug("  Computing harmonic centrality...")
            harmonic = nx.harmonic_centrality(self.G)
            harmonic_undirected = nx.harmonic_centrality(self.U)
            for node in self.G.nodes():
                metrics[node]['harmonic_centrality'] = harmonic[node]
                metrics[node]['harmonic_centrality_undirected'] = harmonic_undirected[node]
            computed.append("harmonic")
        except Exception as e:
            logger.warning(f"  ⚠ Harmonic centrality failed: {e}")

        try:
            logger.debug("  Computing load centrality...")
            load = nx.load_centrality(self.G)
            load_undirected = nx.load_centrality(self.U)
            for node in self.G.nodes():
                metrics[node]['load_centrality'] = load[node]
                metrics[node]['load_centrality_undirected'] = load_undirected[node]
            computed.append("load")
        except Exception as e:
            logger.warning(f"  ⚠ Load centrality failed: {e}")

        try:
            logger.debug("  Computing subgraph centrality...")
            subgraph = nx.subgraph_centrality(self.U)
            for node in self.G.nodes():
                metrics[node]['subgraph_centrality'] = subgraph[node]
            computed.append("subgraph")
        except Exception as e:
            logger.warning(f"  ⚠ Subgraph centrality failed: {e}")

        try:
            if self.n < 1000:
                logger.debug("  Computing second order centrality...")
                second_order = nx.second_order_centrality(self.U)
                for node in self.G.nodes():
                    metrics[node]['second_order_centrality'] = second_order[node]
                computed.append("second_order")
            else:
                logger.debug("  Skipping second order centrality (graph too large)")
        except Exception as e:
            logger.warning(f"  ⚠ Second order centrality failed: {e}")

        try:
            logger.debug("  Computing percolation centrality...")
            import random
            random.seed(42)
            states = {node: random.random() for node in self.G.nodes()}
            percolation = nx.percolation_centrality(self.G, states=states)
            for node in self.G.nodes():
                metrics[node]['percolation_centrality'] = percolation[node]
            computed.append("percolation")
        except Exception as e:
            logger.warning(f"  ⚠ Percolation centrality failed: {e}")

        try:
            logger.debug("  Computing trophic levels...")
            trophic = nx.trophic_levels(self.G)
            for node in self.G.nodes():
                metrics[node]['trophic_level'] = trophic[node]
            computed.append("trophic")
        except Exception as e:
            logger.warning(f"  ⚠ Trophic levels failed: {e}")

        try:
            if nx.is_connected(self.U) and self.n < 1000:
                logger.debug("  Computing current flow centrality...")
                cf_between = nx.current_flow_betweenness_centrality(self.U)
                cf_close = nx.current_flow_closeness_centrality(self.U)
                for node in self.G.nodes():
                    metrics[node]['current_flow_betweenness'] = cf_between[node]
                    metrics[node]['current_flow_closeness'] = cf_close[node]
                computed.append("current_flow")
            else:
                logger.debug("  Skipping current flow (disconnected or too large)")
        except Exception as e:
            logger.warning(f"  ⚠ Current flow centrality failed: {e}")

        try:
            if nx.is_connected(self.U) and self.n < 1000:
                logger.debug("  Computing information centrality...")
                info = nx.information_centrality(self.U)
                for node in self.G.nodes():
                    metrics[node]['information_centrality'] = info[node]
                computed.append("information")
            else:
                logger.debug("  Skipping information centrality (disconnected or too large)")
        except Exception as e:
            logger.warning(f"  ⚠ Information centrality failed: {e}")

        try:
            if self.n < 500:
                logger.debug("  Computing communicability betweenness...")
                comm_between = nx.communicability_betweenness_centrality(self.U)
                for node in self.G.nodes():
                    metrics[node]['communicability_betweenness'] = comm_between[node]
                computed.append("communicability")
            else:
                logger.debug("  Skipping communicability (graph too large)")
        except Exception as e:
            logger.warning(f"  ⚠ Communicability betweenness failed: {e}")

        try:
            logger.debug("  Computing VoteRank...")
            voterank_list = nx.voterank(self.U)
            voterank_dict = {node: len(voterank_list) - i for i, node in enumerate(voterank_list)}
            for node in self.G.nodes():
                metrics[node]['voterank'] = voterank_dict.get(node, 0)
            computed.append("voterank")
        except Exception as e:
            logger.warning(f"  ⚠ VoteRank failed: {e}")

        try:
            logger.debug("  Computing edge betweenness...")
            edge_between = nx.edge_betweenness_centrality(self.G)
            for node in self.G.nodes():
                edges = [(u, v) for u, v in edge_between.keys() if u == node or v == node]
                metrics[node]['edge_betweenness_sum'] = sum(edge_between[e] for e in edges)
            computed.append("edge_betweenness")
        except Exception as e:
            logger.warning(f"  ⚠ Edge betweenness failed: {e}")

        logger.debug(f"  Successfully computed: {', '.join(computed)}")

    def _clustering(self, metrics):
        """Clustering metrics"""
        try:
            logger.debug("  Computing clustering coefficients...")
            clustering = nx.clustering(self.U)
            clustering_dir = nx.clustering(self.G)
            triangles = nx.triangles(self.U)
            triangles_dir = nx.triangles(self.G)
            square = nx.square_clustering(self.U)

            for node in self.G.nodes():
                metrics[node]['clustering_coefficient'] = clustering[node]
                metrics[node]['clustering_coefficient_directed'] = clustering_dir[node]
                metrics[node]['triangle_count'] = triangles[node]
                metrics[node]['triangle_count_directed'] = triangles_dir[node]
                metrics[node]['square_clustering'] = square[node]
                metrics[node]['local_transitivity'] = clustering[node]
            logger.debug(f"  Computed clustering for {len(metrics)} nodes")
        except Exception as e:
            logger.warning(f"  ⚠ Clustering failed: {e}")

    def _community(self, metrics):
        """Community detection"""
        try:
            logger.debug("  Running Louvain community detection...")
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(self.U, threshold=1e-10, seed=42)

            comm_map = {}
            comm_sizes = {}
            for idx, comm in enumerate(communities):
                comm_sizes[idx] = len(comm)
                for node in comm:
                    comm_map[node] = idx

            logger.debug(f"  Found {len(communities)} communities")
            logger.debug(f"  Largest community: {max(comm_sizes.values())} nodes")

            logger.debug("  Computing core numbers...")
            core = nx.core_number(self.U)

            logger.debug("  Computing onion layers...")
            onion = nx.onion_layers(self.U)

            for node in self.G.nodes():
                cid = comm_map.get(node, -1)
                metrics[node]['community_id'] = cid
                metrics[node]['community_size'] = comm_sizes.get(cid, 0)
                metrics[node]['core_number'] = core[node]
                metrics[node]['onion_layer'] = onion[node]

            logger.debug(f"  Max core number: {max(core.values())}")
        except Exception as e:
            logger.warning(f"  ⚠ Community detection failed: {e}")

        try:
            if self.n < 500:
                logger.debug("  Computing local reaching centrality (PARALLEL)...")
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    func = partial(_compute_node_lrc, self.G)
                    futures = {executor.submit(func, node): node for node in self.G.nodes()}

                    for future in as_completed(futures):
                        node, lrc = future.result()
                        metrics[node]['local_reaching_centrality'] = lrc
        except Exception as e:
            logger.warning(f"  ⚠ Local reaching centrality failed: {e}")

    def _paths_parallel(self, metrics):
        """Path metrics with PARALLEL processing"""
        logger.debug(f"  Using {self.n_jobs} workers for parallel path computation...")

        try:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                func = partial(_compute_node_paths, self.G)
                futures = {executor.submit(func, node): node for node in self.G.nodes()}

                completed = 0
                log_interval = max(1, len(futures) // 10)

                for future in as_completed(futures):
                    node, path_metrics = future.result()
                    metrics[node].update(path_metrics)

                    completed += 1
                    if completed % log_interval == 0 or completed == len(futures):
                        pct = 100 * completed / len(futures)
                        logger.debug(f"  Progress: {completed}/{len(futures)} ({pct:.1f}%)")

            logger.debug(f"  Computed paths for {completed} nodes")
        except Exception as e:
            logger.warning(f"  ⚠ Path computation failed: {e}")

        # Eccentricity
        try:
            logger.debug("  Computing eccentricity...")
            if nx.is_connected(self.U):
                ecc = nx.eccentricity(self.U)
                for node in self.G.nodes():
                    metrics[node]['eccentricity'] = ecc[node]
            else:
                logger.debug("  Graph disconnected, using largest component")
                largest_cc = max(nx.connected_components(self.U), key=len)
                subgraph = self.U.subgraph(largest_cc)
                ecc = nx.eccentricity(subgraph)
                for node in self.G.nodes():
                    metrics[node]['eccentricity'] = ecc.get(node, 0)
        except Exception as e:
            logger.warning(f"  ⚠ Eccentricity failed: {e}")

        # Wiener contribution (parallel for small graphs)
        try:
            if self.n < 500:
                logger.debug("  Computing Wiener contributions (PARALLEL)...")
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    func = partial(_compute_node_wiener, self.U)
                    futures = {executor.submit(func, node): node for node in self.G.nodes()}

                    for future in as_completed(futures):
                        node, wiener = future.result()
                        metrics[node]['wiener_contribution'] = wiener
            else:
                logger.debug("  Skipping Wiener (graph too large)")
        except Exception as e:
            logger.warning(f"  ⚠ Wiener contribution failed: {e}")

    def _distances(self, metrics):
        """Distance measures"""
        try:
            logger.debug("  Computing graph diameter and radius...")
            if nx.is_connected(self.U):
                radius = nx.radius(self.U)
                diameter = nx.diameter(self.U)
                center = set(nx.center(self.U))
                periphery = set(nx.periphery(self.U))
            else:
                logger.debug("  Graph disconnected, using largest component")
                largest_cc = max(nx.connected_components(self.U), key=len)
                subgraph = self.U.subgraph(largest_cc)
                radius = nx.radius(subgraph)
                diameter = nx.diameter(subgraph)
                center = set(nx.center(subgraph))
                periphery = set(nx.periphery(subgraph))

            for node in self.G.nodes():
                metrics[node]['graph_radius'] = radius
                metrics[node]['graph_diameter'] = diameter
                metrics[node]['is_center'] = 1 if node in center else 0
                metrics[node]['is_periphery'] = 1 if node in periphery else 0
        except Exception as e:
            logger.warning(f"  ⚠ Distance measures failed: {e}")

    def _structural(self, metrics):
        """Structural metrics"""
        try:
            logger.debug("  Computing structural holes...")
            constraint = nx.constraint(self.U)
            effective = nx.effective_size(self.U)

            for node in self.G.nodes():
                c = constraint.get(node, 0)
                e = effective.get(node, 0)
                metrics[node]['constraint'] = c
                metrics[node]['effective_size'] = e
                deg = self.U.degree(node)
                metrics[node]['redundancy'] = 1 - (e / deg) if deg > 0 else 0
        except Exception as e:
            logger.warning(f"  ⚠ Structural holes failed: {e}")

        try:
            logger.debug("  Computing articulation points and bridges...")
            articulation = set(nx.articulation_points(self.U))
            bridges = set(nx.bridges(self.U))

            logger.debug(f"  Found {len(articulation)} articulation points, {len(bridges)} bridges")

            for node in self.G.nodes():
                metrics[node]['is_articulation_point'] = 1 if node in articulation else 0
                bridge_count = sum(1 for u, v in bridges if u == node or v == node)
                metrics[node]['bridge_count'] = bridge_count
        except Exception as e:
            logger.warning(f"  ⚠ Robustness metrics failed: {e}")

        try:
            logger.debug("  Computing neighbor degree statistics...")
            avg_neighbor_deg_undirected = nx.average_neighbor_degree(self.U)
            avg_neighbor_deg_directed = nx.average_neighbor_degree(self.G)

            for node in self.G.nodes():
                neighbors = list(self.U.neighbors(node))
                if neighbors:
                    degs = [self.U.degree(n) for n in neighbors]
                    metrics[node]['avg_neighbor_degree'] = np.mean(degs)
                    metrics[node]['min_neighbor_degree'] = np.min(degs)
                    metrics[node]['max_neighbor_degree'] = np.max(degs)
                    metrics[node]['std_neighbor_degree'] = np.std(degs)
                else:
                    metrics[node]['avg_neighbor_degree'] = 0
                    metrics[node]['min_neighbor_degree'] = 0
                    metrics[node]['max_neighbor_degree'] = 0
                    metrics[node]['std_neighbor_degree'] = 0

                metrics[node]['avg_neighbor_degree_undirected'] = avg_neighbor_deg_undirected[node]
                metrics[node]['avg_neighbor_degree_directed'] = avg_neighbor_deg_directed[node]
        except Exception as e:
            logger.warning(f"  ⚠ Neighbor degree stats failed: {e}")

    def _reciprocity(self, metrics):
        """Reciprocity metrics"""
        logger.debug("  Computing reciprocity...")
        for node in self.G.nodes():
            out_neighbors = set(self.G.successors(node))
            in_neighbors = set(self.G.predecessors(node))
            mutual = out_neighbors & in_neighbors

            metrics[node]['mutual_count'] = len(mutual)
            metrics[node]['mutual_ratio'] = len(mutual) / len(out_neighbors) if out_neighbors else 0
            metrics[node]['mutual_received_ratio'] = len(mutual) / len(in_neighbors) if in_neighbors else 0
            metrics[node]['one_way_out'] = len(out_neighbors - mutual)
            metrics[node]['one_way_in'] = len(in_neighbors - mutual)

    def _reach_parallel(self, metrics):
        """Reach metrics with PARALLEL processing"""
        logger.debug(f"  Using {self.n_jobs} workers for parallel reach computation...")
        max_hops = 6

        try:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                func = partial(_compute_node_reach, self.G, max_hops=max_hops)
                futures = {executor.submit(func, node): node for node in self.G.nodes()}

                completed = 0
                log_interval = max(1, len(futures) // 10)

                for future in as_completed(futures):
                    node, reach_metrics = future.result()
                    metrics[node].update(reach_metrics)

                    completed += 1
                    if completed % log_interval == 0 or completed == len(futures):
                        pct = 100 * completed / len(futures)
                        logger.debug(f"  Progress: {completed}/{len(futures)} ({pct:.1f}%)")

            logger.debug(f"  Computed reach for {completed} nodes")
        except Exception as e:
            logger.warning(f"  ⚠ Reach computation failed: {e}")

    def _components(self, metrics):
        """Component metrics"""
        try:
            logger.debug("  Computing connected components...")
            weak_comps = list(nx.weakly_connected_components(self.G))
            strong_comps = list(nx.strongly_connected_components(self.G))

            logger.debug(f"  Found {len(weak_comps)} weak components, {len(strong_comps)} strong components")

            weak_map = {}
            weak_sizes = {}
            for idx, comp in enumerate(weak_comps):
                weak_sizes[idx] = len(comp)
                for node in comp:
                    weak_map[node] = idx

            strong_map = {}
            strong_sizes = {}
            for idx, comp in enumerate(strong_comps):
                strong_sizes[idx] = len(comp)
                for node in comp:
                    strong_map[node] = idx

            largest_weak = max(weak_sizes.values()) if weak_sizes else 0
            logger.debug(f"  Largest weak component: {largest_weak} nodes")

            for node in self.G.nodes():
                wid = weak_map.get(node, -1)
                sid = strong_map.get(node, -1)
                metrics[node]['weak_component_size'] = weak_sizes.get(wid, 0)
                metrics[node]['strong_component_size'] = strong_sizes.get(sid, 0)
                metrics[node]['in_largest_component'] = 1 if weak_sizes.get(wid, 0) == largest_weak else 0
        except Exception as e:
            logger.warning(f"  ⚠ Component metrics failed: {e}")

    def _vitality(self, metrics):
        """Vitality metrics"""
        try:
            if nx.is_connected(self.U) and self.n < 500:
                logger.debug("  Computing closeness vitality...")
                vitality = nx.closeness_vitality(self.U)
                for node in self.G.nodes():
                    metrics[node]['closeness_vitality'] = vitality[node]
            else:
                logger.debug("  Skipping vitality (disconnected or too large)")
        except Exception as e:
            logger.warning(f"  ⚠ Vitality failed: {e}")

    def _dispersion(self, metrics):
        """Dispersion metrics"""
        try:
            sample_size = min(100, self.n)
            sampled = list(self.G.nodes())[:sample_size]
            logger.debug(f"  Computing dispersion for {sample_size} sampled nodes...")

            for node in sampled:
                try:
                    disp = nx.dispersion(self.U, node)
                    if disp:
                        metrics[node]['avg_dispersion'] = np.mean(list(disp.values()))
                        metrics[node]['max_dispersion'] = np.max(list(disp.values()))
                    else:
                        metrics[node]['avg_dispersion'] = 0
                        metrics[node]['max_dispersion'] = 0
                except Exception:
                    metrics[node]['avg_dispersion'] = 0
                    metrics[node]['max_dispersion'] = 0
        except Exception as e:
            logger.warning(f"  ⚠ Dispersion failed: {e}")

    def _efficiency(self, metrics):
        """Efficiency metrics"""
        try:
            logger.debug("  Computing local efficiency...")
            local_eff = nx.local_efficiency(self.U)
            for node in self.G.nodes():
                metrics[node]['local_efficiency'] = local_eff
        except Exception as e:
            logger.warning(f"  ⚠ Efficiency failed: {e}")

    def _flow(self, metrics):
        """Flow metrics"""
        try:
            if self.n > 1:
                logger.debug("  Computing flow hierarchy...")
                flow_h = nx.flow_hierarchy(self.G)
                logger.debug(f"  Flow hierarchy: {flow_h:.4f}")
                for node in self.G.nodes():
                    metrics[node]['flow_hierarchy'] = flow_h
        except Exception as e:
            logger.warning(f"  ⚠ Flow hierarchy failed: {e}")

    def _dominance_parallel(self, metrics):
        """Dominance metrics with PARALLEL processing"""
        logger.debug(f"  Using {self.n_jobs} workers for parallel dominance computation...")

        try:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                func = partial(_compute_node_dominance, self.G, n=self.n)
                futures = {executor.submit(func, node): node for node in self.G.nodes()}

                completed = 0
                log_interval = max(1, len(futures) // 10)

                for future in as_completed(futures):
                    node, dom_metrics = future.result()
                    metrics[node].update(dom_metrics)

                    completed += 1
                    if completed % log_interval == 0 or completed == len(futures):
                        pct = 100 * completed / len(futures)
                        logger.debug(f"  Progress: {completed}/{len(futures)} ({pct:.1f}%)")

            logger.debug(f"  Computed dominance for {completed} nodes")
        except Exception as e:
            logger.warning(f"  ⚠ Dominance computation failed: {e}")

    def _safe_alpha(self):
        """Compute safe alpha for Katz centrality"""
        try:
            import scipy.sparse as sp
            adj = nx.to_scipy_sparse_array(self.G, format='csr')
            eigenvals = sp.linalg.eigs(adj, k=1, which='LM', return_eigenvectors=False)
            largest = abs(eigenvals[0])
            alpha = 0.9 / largest if largest > 0 else 0.01
            logger.debug(f"  Computed safe alpha: {alpha:.6f}")
            return alpha
        except Exception:
            logger.debug("  Using fallback alpha: 0.01")
            return 0.01


# ==============================================================================
# DATA LOADING (standalone mode)
# ==============================================================================

def load_data(user, password, host, database):
    """
    Load trust data from PostgreSQL for a SINGLE network (standalone use).

    This is kept simple: one built-in query (CrcV2 trust graph).
    For multi-SQL multi-network mode, stream_to_cytoscape.py has its own loader.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("LOADING DATA FROM DATABASE (standalone)")
    logger.info("=" * 70)
    logger.info(f"Connecting to: {host}/{database}")

    url = URL.create(
        "postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        database=database,
    )

    engine = create_engine(url)

    query = """
    WITH human_avatars AS (
        SELECT avatar 
        FROM public."V_CrcV2_Avatars"
        WHERE "type" = 'CrcV2_RegisterHuman'
    )
    SELECT 
        LOWER(t1.trustee) AS source,
        LOWER(t1.truster) AS target
    FROM "V_CrcV2_TrustRelations" t1
    INNER JOIN human_avatars t2 ON t2.avatar = t1.truster
    INNER JOIN human_avatars t3 ON t3.avatar = t1.trustee
    WHERE t1.truster != t1.trustee
    """

    logger.info("Executing SQL query...")
    start = time.time()
    df = pd.read_sql_query(query, engine)
    elapsed = time.time() - start

    engine.dispose()

    logger.info(f"✓ Loaded {len(df):,} trust relations in {elapsed:.2f}s")
    logger.info(f"  • Unique sources: {df['source'].nunique():,}")
    logger.info(f"  • Unique targets: {df['target'].nunique():,}")
    logger.info("=" * 70)

    return df


# ==============================================================================
# HELP AND INFO FUNCTIONS
# ==============================================================================

def print_help():
    """Print help information about available metrics modes"""
    print("\n" + "=" * 70)
    print("GRAPH METRICS CALCULATOR - HELP")
    print("=" * 70)
    print("\nUsage: python graph_metrics.py")
    print("\nMETRICS MODES")
    print("-" * 40)
    print("\nSet METRICS_MODE in your .env file to control which metrics to compute.")
    print("\nAvailable Presets:")
    for preset, categories in METRIC_PRESETS.items():
        print(f"\n  {preset:10} - Computes {len(categories)} categories:")
        if len(categories) <= 6:
            for cat in categories:
                print(f"               • {cat}")
        else:
            print(f"               • {', '.join(categories[:3])},")
            print(f"               • {', '.join(categories[3:6])},")
            print(f"               • ... and {len(categories) - 6} more")

    print("\nIndividual Categories:")
    for cat, desc in METRIC_CATEGORIES.items():
        print(f"  {cat:12} - {desc}")

    print("\nCustom Selection:")
    print("  You can also use comma-separated categories, e.g.:")
    print("  METRICS_MODE=topology,clustering,community")

    print("\nExamples:")
    print("-" * 40)
    print("  METRICS_MODE=basic        # Just topology and clustering")
    print("  METRICS_MODE=essential    # Most important metrics")
    print("  METRICS_MODE=all          # Compute everything (default)")
    print("  METRICS_MODE=topology     # Just basic topology")
    print("  METRICS_MODE=topology,centrality,clustering  # Custom selection")

    print("\nPerformance Tips:")
    print("-" * 40)
    print("  • 'basic' mode: ~2-5 seconds for most graphs")
    print("  • 'essential' mode: ~1-5 minutes")
    print("  • 'moderate' mode: ~5-15 minutes")
    print("  • 'all' mode: ~15-60+ minutes (depends on graph size)")
    print("\n" + "=" * 70 + "\n")


# ==============================================================================
# MAIN EXECUTION (standalone CLI)
# ==============================================================================

def main():
    overall_start = time.time()

    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_help()
        sys.exit(0)

    logger.info("")
    logger.info("#" * 70)
    logger.info("#" + " " * 68 + "#")
    logger.info("#" + "  GRAPH METRICS CALCULATOR - PARALLEL VERSION".center(68) + "#")
    logger.info("#" + " " * 68 + "#")
    logger.info("#" * 70)

    # Load config
    logger.info("")
    logger.info("Loading configuration from .env file...")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    OUTPUT = os.getenv("OUTPUT_FILE", "graph_metrics.csv")
    N_JOBS = int(os.getenv("N_JOBS", "2")) or None
    METRICS_MODE = os.getenv("METRICS_MODE", "all")

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        logger.error("✗ Missing required DB credentials in .env file!")
        logger.error("  Required: DB_USER, DB_PASSWORD, DB_HOST, DB_NAME")
        raise ValueError("Missing DB credentials in .env file")

    logger.info("✓ Configuration loaded successfully")
    logger.info(f"  • Output file: {OUTPUT}")
    logger.info(f"  • Metrics mode: {METRICS_MODE}")
    if N_JOBS:
        logger.info(f"  • Parallel workers: {N_JOBS}")

    # Load data
    df_edges = load_data(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)

    # Build graph
    logger.info("")
    logger.info("=" * 70)
    logger.info("BUILDING GRAPH")
    logger.info("=" * 70)
    logger.info("Creating directed graph from edges...")

    start = time.time()
    G = nx.DiGraph()
    for _, row in df_edges.iterrows():
        G.add_edge(row['source'], row['target'])
    elapsed = time.time() - start

    logger.info(f"✓ Graph built in {elapsed:.2f}s")
    logger.info("=" * 70)

    # Compute metrics
    calculator = GraphMetrics(G, n_jobs=N_JOBS, metrics_mode=METRICS_MODE)
    df_metrics = calculator.compute_all()

    # Save
    logger.info("")
    logger.info("=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)
    logger.info(f"Writing to: {OUTPUT}")

    start = time.time()
    df_metrics.to_csv(OUTPUT, index=False)
    elapsed = time.time() - start

    logger.info(f"✓ Saved {len(df_metrics):,} rows x {len(df_metrics.columns)} columns in {elapsed:.2f}s")

    # Final summary
    overall_elapsed = time.time() - overall_start
    logger.info("")
    logger.info("#" * 70)
    logger.info("#" + " " * 68 + "#")
    logger.info("#" + "  EXECUTION COMPLETE".center(68) + "#")
    logger.info("#" + " " * 68 + "#")
    logger.info("#" * 70)
    logger.info("")
    logger.info(f"Total execution time: {overall_elapsed:.2f}s ({overall_elapsed / 60:.1f} minutes)")
    logger.info(f"Output file: {OUTPUT}")
    logger.info(f"Nodes processed: {len(df_metrics):,}")
    logger.info(f"Metrics computed: {len(df_metrics.columns) - 1}")
    logger.info(f"Time per node: {overall_elapsed / len(df_metrics):.4f}s")
    logger.info("")
    logger.info("To see available metrics modes, run: python graph_metrics.py --help")
    logger.info("")
    logger.info("#" * 70)


if __name__ == "__main__":
    main()
