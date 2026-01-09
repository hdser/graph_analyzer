"""
Capacity Flow Service

Service layer for capacity graph and max flow operations.
Uses the SELECTED graph from network_service.

Location: web_viewer/backend/services/capacity_flow_service.py
"""
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

from ..config import settings

logger = logging.getLogger(__name__)

# Import capacity flow engine
HAS_CAPACITY_FLOW = False
CapacityFlowEngine = None
TrustRelation = None
TokenBalance = None
FlowResult = None
CapacityGraphStats = None

def get_available_backends():
    return []

def get_backend_info():
    return []

try:
    from engines.capacity_flow import (
        CapacityFlowEngine,
        TrustRelation,
        TokenBalance,
        FlowResult,
        CapacityGraphStats,
        get_available_backends,
        get_backend_info,
    )
    HAS_CAPACITY_FLOW = True
    logger.info("Capacity flow engine loaded successfully")
except ImportError:
    try:
        service_dir = Path(__file__).parent
        backend_dir = service_dir.parent
        web_viewer_dir = backend_dir.parent
        
        if str(web_viewer_dir) not in sys.path:
            sys.path.insert(0, str(web_viewer_dir))
        
        from engines.capacity_flow import (
            CapacityFlowEngine,
            TrustRelation,
            TokenBalance,
            FlowResult,
            CapacityGraphStats,
            get_available_backends,
            get_backend_info,
        )
        HAS_CAPACITY_FLOW = True
        logger.info("Capacity flow engine loaded (after path fix)")
    except ImportError as e:
        logger.warning(f"Capacity flow engine not available: {e}")


class CapacityFlowService:
    """
    Service for capacity graph and max flow operations.
    Uses the SELECTED graph from network_service (not all graphs).
    """
    
    DEFAULT_ROUTER_ADDRESS = "0xdc287474114cc0551a81ddc2eb51783fbf34802f"
    
    def __init__(self):
        self._engines: Dict[str, Any] = {}
        self._network_service = None
        self._last_build_stats: Dict[str, Dict] = {}
        
    def set_network_service(self, network_service) -> None:
        self._network_service = network_service
    
    def _get_network_service(self):
        if self._network_service is None:
            try:
                from .network_service import network_service
                self._network_service = network_service
            except ImportError:
                pass
        return self._network_service
    
    def is_available(self) -> bool:
        return HAS_CAPACITY_FLOW
    
    def get_engine(self, graph_id: Optional[str] = None) -> Optional[Any]:
        if not HAS_CAPACITY_FLOW:
            return None
        
        graph_id = graph_id or "default"
        
        if graph_id not in self._engines:
            router = getattr(settings, 'CIRCLES_ROUTER_ADDRESS', self.DEFAULT_ROUTER_ADDRESS)
            self._engines[graph_id] = CapacityFlowEngine(router_address=router)
        
        return self._engines[graph_id]
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        if not HAS_CAPACITY_FLOW:
            return []
        return get_backend_info()
    
    def get_algorithms(self) -> List[Dict[str, Any]]:
        """Get all available algorithms."""
        algorithms = [
            {
                "id": "networkx:edmonds_karp",
                "label": "Edmonds-Karp (NetworkX)",
                "backend": "networkx",
                "algorithm": "edmonds_karp",
                "supports_cutoff": True,
                "description": "BFS-based, supports cutoff"
            },
            {
                "id": "networkx:shortest_augmenting_path",
                "label": "Shortest Augmenting Path (NetworkX)",
                "backend": "networkx",
                "algorithm": "shortest_augmenting_path",
                "supports_cutoff": True,
                "description": "Similar to Edmonds-Karp"
            },
            {
                "id": "networkx:preflow_push",
                "label": "Preflow Push (NetworkX)",
                "backend": "networkx",
                "algorithm": "preflow_push",
                "supports_cutoff": False,
                "description": "Fastest for dense graphs"
            },
            {
                "id": "networkx:dinitz",
                "label": "Dinitz (NetworkX)",
                "backend": "networkx",
                "algorithm": "dinitz",
                "supports_cutoff": False,
                "description": "Good general performance"
            },
        ]
        
        # Add OR-Tools if available
        try:
            from engines.capacity_flow.backends.ortools_backend import ORToolsBackend
            if ORToolsBackend.is_available():
                algorithms.append({
                    "id": "ortools:max_flow",
                    "label": "Max Flow (OR-Tools)",
                    "backend": "ortools",
                    "algorithm": "max_flow",
                    "supports_cutoff": False,
                    "description": "Google OR-Tools solver"
                })
        except ImportError:
            pass
        
        return algorithms
    
    def get_stats(self, graph_id: Optional[str] = None) -> Dict[str, Any]:
        engine = self.get_engine(graph_id)
        if engine is None:
            return {"error": "Capacity flow not available"}
        
        stats = engine.get_stats()
        if stats is None:
            return {"built": False, "message": "Capacity graph not built"}
        
        return {
            "built": True,
            "stats": stats.to_dict(),
            "build_time": engine.get_build_time()
        }
    
    def _get_target_graph_id(self, ns, graph_id: Optional[str] = None) -> str:
        """
        Get the target graph ID to use.
        
        Priority:
        1. Explicit graph_id parameter
        2. Graph with 'trust' in name (preferred for capacity flow)
        3. First available graph
        """
        if graph_id and graph_id in ns.graphs:
            return graph_id
        
        # Look for a trust graph
        for gid in ns.graphs.keys():
            if 'trust' in gid.lower() and 'invite' not in gid.lower():
                return gid
        
        # Fallback to first graph
        if ns.graphs:
            return list(ns.graphs.keys())[0]
        
        return None
    
    async def build_capacity_graph(
        self,
        graph_id: Optional[str] = None,
        include_groups: bool = True,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Build capacity graph from the SELECTED graph.
        Only uses one graph - the trusts graph if available.
        """
        if not HAS_CAPACITY_FLOW:
            return {"success": False, "error": "Capacity flow engine not available"}
        
        engine = self.get_engine(graph_id)
        
        if not force_rebuild and engine.is_ready():
            stats = engine.get_stats()
            # Return cached stats with previously stored counts
            cached_build = self._last_build_stats.get(graph_id or "default", {})
            return {
                "success": True,
                "cached": True,
                "stats": stats.to_dict() if stats else {},
                "build_time": engine.get_build_time(),
                "trust_count": cached_build.get("trust_count", 0),
                "balance_count": cached_build.get("balance_count", 0),
                "group_count": cached_build.get("group_count", 0),
                "source_graph": cached_build.get("source_graph", "")
            }
        
        ns = self._get_network_service()
        if ns is None:
            return {"success": False, "error": "Network service not available"}
        
        if not ns.graphs:
            return {"success": False, "error": "No graphs loaded. Load a graph first."}
        
        start_time = time.time()
        
        try:
            # Get the target graph (trusts if available)
            target_graph_id = self._get_target_graph_id(ns, graph_id)
            if not target_graph_id:
                return {"success": False, "error": "No suitable graph found"}
            
            print(f"[CapacityFlow] Using graph: {target_graph_id}")
            print(f"[CapacityFlow] Available graphs: {list(ns.graphs.keys())}")
            
            G = ns.graphs[target_graph_id]
            df = ns.edge_layers.get(target_graph_id)
            
            print(f"[CapacityFlow] Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            if df is not None:
                print(f"[CapacityFlow] Edge layer has {len(df)} rows, columns: {list(df.columns)}")
            
            # Extract trusts from this specific graph
            trusts = self._extract_trusts_from_graph(G, df)
            print(f"[CapacityFlow] Extracted {len(trusts)} trust relations")
            
            # Extract balances from node properties 'tokens' field
            balances = self._extract_balances_from_graph(G)
            print(f"[CapacityFlow] Extracted {len(balances)} token balances")
            
            # Get groups
            groups = self._get_groups_from_graph(G) if include_groups else set()
            print(f"[CapacityFlow] Found {len(groups)} groups")
            
            # Create default balances if none found
            if not balances:
                print(f"[CapacityFlow] No balances found, creating default self-token balances")
                balances = self._create_default_balances(trusts)
            
            # Build the capacity graph
            stats = engine.build_graph(
                trusts=trusts,
                balances=balances,
                groups=groups
            )
            
            build_time = time.time() - start_time
            
            self._last_build_stats[graph_id or "default"] = {
                "trust_count": len(trusts),
                "balance_count": len(balances),
                "group_count": len(groups),
                "source_graph": target_graph_id,
            }
            
            return {
                "success": True,
                "cached": False,
                "stats": stats.to_dict(),
                "build_time": build_time,
                "trust_count": len(trusts),
                "balance_count": len(balances),
                "group_count": len(groups),
                "source_graph": target_graph_id
            }
            
        except Exception as e:
            logger.error(f"Failed to build capacity graph: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _extract_trusts_from_graph(self, G: nx.DiGraph, df: Optional[pd.DataFrame]) -> List[Any]:
        """
        Extract trust relations from a NetworkX graph.
        Each edge is a trust relation: truster -> trustee
        """
        trusts = []
        
        for u, v in G.edges():
            truster = str(u).lower()
            trustee = str(v).lower()
            
            if truster in ('nan', 'none', '') or trustee in ('nan', 'none', ''):
                continue
            
            trust = TrustRelation(
                truster=truster,
                trustee=trustee,
                limit=0,
                expiry=None
            )
            trusts.append(trust)
        
        return trusts
    
    def _extract_balances_from_graph(self, G: nx.DiGraph) -> List[Any]:
        """
        Extract token balances from node properties.
        
        The data format is:
        - tokens: list of token addresses (strings)
        - tokens_balance: list of balances (parallel to tokens)
        - OR total_balance: single total balance
        - OR supply: the avatar's own token supply
        """
        balances = []
        seen = set()
        
        # Debug: check first few nodes
        sample_count = 0
        for node in G.nodes():
            node_data = G.nodes[node]
            node_addr = str(node).lower()
            
            # Debug first 3 nodes
            if sample_count < 3:
                print(f"[CapacityFlow] DEBUG node {node_addr[:10]}...")
                for key in ['tokens', 'tokens_balance', 'total_balance', 'supply']:
                    if key in node_data:
                        val = node_data[key]
                        print(f"[CapacityFlow]   {key}: type={type(val).__name__}, value={str(val)[:200]}")
                sample_count += 1
            
            # Method 1: tokens + tokens_balance as parallel lists
            tokens = node_data.get('tokens')
            tokens_balance = node_data.get('tokens_balance')
            
            if tokens is not None and tokens_balance is not None:
                # Convert numpy arrays to lists
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.tolist()
                if isinstance(tokens_balance, np.ndarray):
                    tokens_balance = tokens_balance.tolist()
                
                if isinstance(tokens, list) and isinstance(tokens_balance, list):
                    # Parallel lists of token addresses and balances
                    for i, token_addr in enumerate(tokens):
                        if i >= len(tokens_balance):
                            break
                        
                        token_id = str(token_addr).lower()
                        try:
                            balance = int(float(tokens_balance[i]))
                        except (ValueError, TypeError, IndexError):
                            balance = 0
                        
                        if balance > 0:
                            key = (node_addr, token_id)
                            if key not in seen:
                                seen.add(key)
                                balances.append(TokenBalance(
                                    holder=node_addr,
                                    token_id=token_id,
                                    balance=balance
                                ))
            
            # Method 2: Use 'supply' as the avatar's own token balance
            # In Circles, each avatar mints their own token (token_id = avatar address)
            supply = node_data.get('supply')
            if supply is not None:
                try:
                    supply_val = int(float(supply))
                    if supply_val > 0:
                        key = (node_addr, node_addr)  # holder holds their own token
                        if key not in seen:
                            seen.add(key)
                            balances.append(TokenBalance(
                                holder=node_addr,
                                token_id=node_addr,  # token_id = avatar address
                                balance=supply_val
                            ))
                except (ValueError, TypeError):
                    pass
            
            # Method 3: If tokens is a list of addresses and no balance info,
            # assume the holder has some balance of each (use default)
            if tokens is not None and tokens_balance is None:
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.tolist()
                
                if isinstance(tokens, list):
                    for token_addr in tokens:
                        if isinstance(token_addr, str):
                            token_id = token_addr.lower()
                            key = (node_addr, token_id)
                            if key not in seen:
                                seen.add(key)
                                # Use a default balance since we don't have actual amounts
                                balances.append(TokenBalance(
                                    holder=node_addr,
                                    token_id=token_id,
                                    balance=10**18  # Default 1 token
                                ))
        
        return balances
    
    def _parse_tokens_field(self, tokens_data: Any, holder: str, seen: Set) -> List[Any]:
        """
        Parse tokens from various formats.
        
        Expected formats:
        - List of dicts: [{"tokenAddress": "0x...", "balance": 1000}, ...]
        - numpy array of dicts
        - Could also be a JSON string
        """
        balances = []
        
        # Handle None
        if tokens_data is None:
            return balances
        
        # Handle string (might be JSON)
        if isinstance(tokens_data, str):
            try:
                import json
                tokens_data = json.loads(tokens_data)
            except (json.JSONDecodeError, TypeError):
                return balances
        
        # Handle numpy arrays
        if isinstance(tokens_data, np.ndarray):
            tokens_data = tokens_data.tolist()
        
        # Must be a list/tuple at this point
        if not isinstance(tokens_data, (list, tuple)):
            return balances
        
        for token_info in tokens_data:
            # Handle nested numpy array
            if isinstance(token_info, np.ndarray):
                token_info = token_info.tolist()
            
            # Must be a dict
            if not isinstance(token_info, dict):
                continue
            
            # Get token address - try many possible field names
            token_id = None
            for key in ['tokenAddress', 'token_address', 'token_id', 'tokenId', 'token', 'id', 'address']:
                if key in token_info:
                    token_id = token_info[key]
                    break
            
            if not token_id:
                continue
            token_id = str(token_id).lower()
            
            # Get balance - try many possible field names
            balance_val = 0
            for key in ['balance', 'amount', 'value', 'qty', 'quantity']:
                if key in token_info and token_info[key]:
                    balance_val = token_info[key]
                    break
            
            try:
                # Handle string numbers and scientific notation
                if isinstance(balance_val, str):
                    balance_val = float(balance_val)
                balance = int(float(balance_val))
            except (ValueError, TypeError):
                balance = 0
            
            if balance > 0:
                key = (holder, token_id)
                if key not in seen:
                    seen.add(key)
                    balances.append(TokenBalance(
                        holder=holder,
                        token_id=token_id,
                        balance=balance
                    ))
        
        return balances
    
    def _get_groups_from_graph(self, G: nx.DiGraph) -> Set[str]:
        """Get group addresses from node properties."""
        groups = set()
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # Check 'type' field
            node_type = node_data.get('type', '')
            if isinstance(node_type, str) and node_type.lower() == 'group':
                groups.add(str(node).lower())
            
            # Check 'isGroup' boolean
            if node_data.get('isGroup') == True:
                groups.add(str(node).lower())
        
        return groups
    
    def _create_default_balances(self, trusts: List[Any]) -> List[Any]:
        """Create default balances: each avatar has their own token."""
        balances = []
        avatars = set()
        
        for trust in trusts:
            avatars.add(trust.truster)
            avatars.add(trust.trustee)
        
        DEFAULT_BALANCE = 10**18
        
        for avatar in avatars:
            balances.append(TokenBalance(
                holder=avatar,
                token_id=avatar,
                balance=DEFAULT_BALANCE
            ))
        
        return balances
    
    async def compute_max_flow(
        self,
        source: str,
        sink: str,
        graph_id: Optional[str] = None,
        backend: Optional[str] = None,
        algorithm: Optional[str] = None,
        cutoff: Optional[int] = None,
        decompose_paths: bool = True,
        simplify_paths: bool = True,
        max_paths: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute maximum flow between source and sink."""
        if not HAS_CAPACITY_FLOW:
            return {"success": False, "error": "Capacity flow engine not available"}
        
        engine = self.get_engine(graph_id)
        
        # Auto-build if needed
        if not engine.is_ready():
            build_result = await self.build_capacity_graph(graph_id)
            if not build_result.get("success"):
                return {
                    "success": False,
                    "error": f"Failed to build graph: {build_result.get('error')}"
                }
        
        try:
            result = engine.compute_max_flow(
                source=source.lower(),
                sink=sink.lower(),
                backend=backend,
                algorithm=algorithm,
                cutoff=cutoff,
                decompose_paths=decompose_paths,
                simplify_paths=simplify_paths,
                max_paths=max_paths
            )
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Max flow failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "source": source, "sink": sink}
    
    def get_capacity_graph_nodes(
        self,
        graph_id: Optional[str] = None,
        use_trust_layout: bool = True
    ) -> Dict[str, Any]:
        """Get capacity graph nodes with positions and original node properties."""
        engine = self.get_engine(graph_id)
        if engine is None or not engine.is_ready():
            return {"error": "Capacity graph not built"}
        
        try:
            graph = engine.get_graph()
            mapper = engine.get_mapper()
            
            if graph is None:
                return {"error": "No graph available"}
            
            # Get the original trust graph for node properties
            ns = self._get_network_service()
            original_graph = None
            original_node_props = {}
            
            if ns:
                # Get properties from source graph
                build_stats = self._last_build_stats.get(graph_id or "default", {})
                source_graph_id = build_stats.get("source_graph")
                
                if source_graph_id and source_graph_id in ns.graphs:
                    original_graph = ns.graphs[source_graph_id]
                    # Build lookup of original node properties by address
                    for node_id in original_graph.nodes():
                        node_data = original_graph.nodes[node_id]
                        addr = node_id.lower()
                        original_node_props[addr] = dict(node_data)
            
            # Get positions from trust graph layout
            positions = {}
            if use_trust_layout and ns and ns.layouts:
                print(f"[CapacityFlow] Available layouts: {list(ns.layouts.keys())}")
                
                # Prioritize trust layout
                layout_priority = [
                    'crc_v2_trusts', 'trusts', 
                ]
                
                selected_layout = None
                
                # First try priority layouts
                for layout_name in layout_priority:
                    if layout_name in ns.layouts and ns.layouts[layout_name]:
                        positions = ns.layouts[layout_name]
                        selected_layout = layout_name
                        break
                
                # If not found, try any layout with 'trust' in name
                if not selected_layout:
                    for layout_name, layout_data in ns.layouts.items():
                        if 'trust' in layout_name.lower() and layout_data:
                            positions = layout_data
                            selected_layout = layout_name
                            break
                
                # Last resort: first non-empty layout
                if not selected_layout:
                    for layout_name, layout_data in ns.layouts.items():
                        if layout_data:
                            positions = layout_data
                            selected_layout = layout_name
                            break
                
                if selected_layout:
                    print(f"[CapacityFlow] Using layout '{selected_layout}' with {len(positions)} positions")
                    sample_keys = list(positions.keys())[:5]
                    print(f"[CapacityFlow] Sample position keys: {sample_keys}")
            
            nodes = []
            with_positions = 0
            without_positions = 0
            
            for node_id, node in graph.nodes.items():
                address = node.address
                if mapper and not address:
                    address = mapper.get_address(node_id)
                
                node_data = {
                    "data": {
                        "id": node_id,
                        "type": node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                        "label": self._short_address(address) if address else node_id,
                    }
                }
                
                if address:
                    node_data["data"]["address"] = address
                    
                    # Include original node properties from trust graph
                    addr_lower = address.lower()
                    if addr_lower in original_node_props:
                        orig_props = original_node_props[addr_lower]
                        for key, value in orig_props.items():
                            # Skip keys we already have or that shouldn't be copied
                            if key in ['id', 'label', 'position', 'x', 'y']:
                                continue
                            # Handle numpy arrays
                            if hasattr(value, 'tolist'):
                                node_data["data"][key] = value.tolist()
                            elif value is not None:
                                node_data["data"][key] = value
                    
                    # Try multiple position lookups
                    pos = None
                    if address in positions:
                        pos = positions[address]
                    elif address.lower() in positions:
                        pos = positions[address.lower()]
                    
                    if pos:
                        node_data["position"] = pos
                        with_positions += 1
                    else:
                        without_positions += 1
                else:
                    without_positions += 1
                
                if node.token_id:
                    node_data["data"]["token_id"] = node.token_id
                
                nodes.append(node_data)
            
            print(f"[CapacityFlow] Nodes with positions: {with_positions}, without: {without_positions}")
            
            stats = graph.get_stats()
            
            return {
                "success": True,
                "nodes": nodes,
                "node_count": len(nodes),
                "edge_count": stats.num_edges if stats else 0,
                "has_positions": with_positions > 0,
                "positions_count": with_positions,
                "stats": stats.to_dict() if stats else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get capacity graph nodes: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def get_capacity_graph_edges(
        self,
        graph_id: Optional[str] = None,
        offset: int = 0,
        limit: int = 10000,
        edge_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get capacity graph edges in batches."""
        engine = self.get_engine(graph_id)
        if engine is None or not engine.is_ready():
            return {"error": "Capacity graph not built"}
        
        try:
            graph = engine.get_graph()
            if graph is None:
                return {"error": "No graph available"}
            
            all_edges = graph.edges
            
            if edge_type:
                all_edges = [
                    e for e in all_edges 
                    if (e.edge_type.value if hasattr(e.edge_type, 'value') else str(e.edge_type)) == edge_type
                ]
            
            total_edges = len(all_edges)
            batch_edges = all_edges[offset:offset + limit]
            
            edges = []
            for i, edge in enumerate(batch_edges):
                edge_data = {
                    "data": {
                        "id": f"e{offset + i}",
                        "source": edge.source,
                        "target": edge.target,
                        "capacity": edge.capacity,
                        "type": edge.edge_type.value if hasattr(edge.edge_type, 'value') else str(edge.edge_type),
                    }
                }
                if edge.token_id:
                    edge_data["data"]["token_id"] = edge.token_id
                edges.append(edge_data)
            
            return {
                "success": True,
                "edges": edges,
                "offset": offset,
                "limit": limit,
                "returned": len(edges),
                "total": total_edges,
                "has_more": offset + len(edges) < total_edges
            }
            
        except Exception as e:
            logger.error(f"Failed to get capacity graph edges: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def get_capacity_graph_data(self, graph_id: Optional[str] = None) -> Dict[str, Any]:
        """Get full capacity graph data."""
        engine = self.get_engine(graph_id)
        if engine is None or not engine.is_ready():
            return {"error": "Capacity graph not built"}
        
        try:
            graph = engine.get_graph()
            mapper = engine.get_mapper()
            
            if graph is None:
                return {"error": "No graph available"}
            
            nodes = []
            edges = []
            
            for node_id, node in graph.nodes.items():
                address = node.address
                if mapper and not address:
                    address = mapper.get_address(node_id)
                
                node_data = {
                    "data": {
                        "id": node_id,
                        "type": node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                        "label": self._short_address(address) if address else node_id,
                    }
                }
                if address:
                    node_data["data"]["address"] = address
                if node.token_id:
                    node_data["data"]["token_id"] = node.token_id
                nodes.append(node_data)
            
            for i, edge in enumerate(graph.edges):
                edge_data = {
                    "data": {
                        "id": f"e{i}",
                        "source": edge.source,
                        "target": edge.target,
                        "capacity": edge.capacity,
                        "type": edge.edge_type.value if hasattr(edge.edge_type, 'value') else str(edge.edge_type),
                    }
                }
                if edge.token_id:
                    edge_data["data"]["token_id"] = edge.token_id
                edges.append(edge_data)
            
            return {
                "success": True,
                "nodes": nodes,
                "edges": edges,
                "stats": graph.get_stats().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to get capacity graph data: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _short_address(self, addr: str) -> str:
        if not addr or len(addr) < 10:
            return addr or ""
        return f"{addr[:6]}...{addr[-4:]}"
    
    def get_node_capacity(
        self,
        address: str,
        direction: str = "both",
        graph_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not HAS_CAPACITY_FLOW:
            return {"error": "Capacity flow not available"}
        
        engine = self.get_engine(graph_id)
        if not engine.is_ready():
            return {"error": "Graph not built"}
        
        return engine.get_node_capacity(address.lower(), direction)
    
    def clear(self, graph_id: Optional[str] = None) -> Dict[str, Any]:
        if graph_id:
            if graph_id in self._engines:
                self._engines[graph_id].clear()
                del self._engines[graph_id]
            if graph_id in self._last_build_stats:
                del self._last_build_stats[graph_id]
            return {"cleared": graph_id}
        else:
            for engine in self._engines.values():
                engine.clear()
            self._engines.clear()
            self._last_build_stats.clear()
            return {"cleared": "all"}


# Singleton
capacity_flow_service = CapacityFlowService()