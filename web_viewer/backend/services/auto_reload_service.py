"""
Auto-Reload Service

Background service for automatic data refresh with SSE notifications.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from threading import Lock

from backend.config import settings, HAS_SSE
from backend.models.requests import AutoReloadConfig
from backend.models.responses import AutoReloadStatus


class AutoReloadManager:
    """
    Manages automatic background reloading of network data.
    
    Features:
    - Configurable reload interval (60-3600 seconds)
    - SSE-based event broadcasting
    - Atomic state updates with diff computation
    - Thread-safe operation
    """
    
    def __init__(self):
        """Initialize the auto-reload manager."""
        self._enabled = False
        self._interval_seconds = settings.AUTO_RELOAD_DEFAULT_INTERVAL
        self._sql_files: List[str] = []
        self._compute_metrics = True
        self._metrics_mode = "essential"
        
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        self._last_reload_time: Optional[datetime] = None
        self._next_reload_time: Optional[datetime] = None
        self._reload_in_progress = False
        self._current_node_count = 0
        self._last_reload_duration: Optional[float] = None
        self._last_reload_nodes_added = 0
        self._last_reload_nodes_removed = 0
        self._error: Optional[str] = None
        
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = Lock()
        
        # Reference to network service (set by network_service on init)
        self._network_service = None
    
    def set_network_service(self, service):
        """Set reference to network service for reload operations."""
        self._network_service = service
    
    def get_status(self) -> AutoReloadStatus:
        """Get current auto-reload status."""
        return AutoReloadStatus(
            enabled=self._enabled,
            interval_seconds=self._interval_seconds,
            last_reload_time=self._last_reload_time,
            next_reload_time=self._next_reload_time,
            reload_in_progress=self._reload_in_progress,
            current_node_count=self._current_node_count,
            last_reload_duration=self._last_reload_duration,
            last_reload_nodes_added=self._last_reload_nodes_added,
            last_reload_nodes_removed=self._last_reload_nodes_removed,
            error=self._error
        )
    
    async def start(self, config: AutoReloadConfig) -> AutoReloadStatus:
        """
        Start auto-reload with given configuration.
        
        Args:
            config: Auto-reload configuration
            
        Returns:
            Current status
        """
        # Stop existing task if running
        await self.stop()
        
        # Update configuration
        self._enabled = config.enabled
        self._interval_seconds = max(
            settings.AUTO_RELOAD_MIN_INTERVAL,
            min(settings.AUTO_RELOAD_MAX_INTERVAL, config.interval_seconds)
        )
        if config.sql_files:
            self._sql_files = config.sql_files
        self._compute_metrics = config.compute_metrics
        self._metrics_mode = config.metrics_mode
        
        if self._enabled:
            self._stop_event.clear()
            self._task = asyncio.create_task(self._reload_loop())
            self._next_reload_time = datetime.now() + timedelta(seconds=self._interval_seconds)
            print(f"[AUTO-RELOAD] Started with interval {self._interval_seconds}s")
            
            # Broadcast status
            await self._broadcast_event("status_update", self.get_status().model_dump())
        
        return self.get_status()
    
    async def stop(self) -> AutoReloadStatus:
        """
        Stop auto-reload.
        
        Returns:
            Current status
        """
        self._enabled = False
        self._stop_event.set()
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        self._next_reload_time = None
        print("[AUTO-RELOAD] Stopped")
        
        # Broadcast status
        await self._broadcast_event("status_update", self.get_status().model_dump())
        
        return self.get_status()
    
    def subscribe(self) -> asyncio.Queue:
        """
        Subscribe to reload events.
        
        Returns:
            Queue for receiving events
        """
        queue = asyncio.Queue()
        with self._lock:
            self._subscribers.add(queue)
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue):
        """
        Unsubscribe from reload events.
        
        Args:
            queue: Queue to unsubscribe
        """
        with self._lock:
            self._subscribers.discard(queue)
    
    async def _broadcast_event(self, event_type: str, data: Any):
        """Broadcast event to all subscribers."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        with self._lock:
            dead_queues = []
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    dead_queues.append(queue)
            
            for queue in dead_queues:
                self._subscribers.discard(queue)
    
    async def _reload_loop(self):
        """Background loop for periodic reloading."""
        while self._enabled and not self._stop_event.is_set():
            try:
                # Wait for interval
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval_seconds
                )
                # Stop event was set
                break
            except asyncio.TimeoutError:
                # Interval elapsed, perform reload
                await self._perform_reload()
                
                # Update next reload time
                self._next_reload_time = datetime.now() + timedelta(seconds=self._interval_seconds)
    
    async def _perform_reload(self):
        """Perform a single reload operation."""
        if not self._network_service:
            print("[AUTO-RELOAD] No network service configured")
            return
        
        self._reload_in_progress = True
        self._error = None
        start_time = time.time()
        
        # Broadcast start event
        await self._broadcast_event("reload_started", {
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # Get current node set for diff
            old_nodes = set()
            for gid, G in self._network_service.graphs.items():
                old_nodes.update(G.nodes())
            
            # Perform reload
            from backend.models.requests import LoadConfig
            config = LoadConfig(
                sql_files=self._sql_files,
                use_cached_layout=True,
                skip_sql=False,
                metrics_mode=self._metrics_mode if self._compute_metrics else "minimal"
            )
            
            result = self._network_service.load_network(config)
            
            # Get new node set for diff
            new_nodes = set()
            for gid, G in self._network_service.graphs.items():
                new_nodes.update(G.nodes())
            
            # Compute diff
            nodes_added = new_nodes - old_nodes
            nodes_removed = old_nodes - new_nodes
            
            # Update stats
            self._last_reload_time = datetime.now()
            self._last_reload_duration = time.time() - start_time
            self._current_node_count = result.node_count
            self._last_reload_nodes_added = len(nodes_added)
            self._last_reload_nodes_removed = len(nodes_removed)
            
            print(f"[AUTO-RELOAD] Complete: {result.node_count} nodes "
                  f"(+{len(nodes_added)}/-{len(nodes_removed)}) in {self._last_reload_duration:.1f}s")
            
            # Broadcast complete event
            await self._broadcast_event("reload_complete", {
                "timestamp": datetime.now().isoformat(),
                "duration": self._last_reload_duration,
                "node_count": result.node_count,
                "nodes_added": len(nodes_added),
                "nodes_removed": len(nodes_removed),
                "added_ids": list(nodes_added)[:100],  # Limit for large changes
                "removed_ids": list(nodes_removed)[:100]
            })
            
        except Exception as e:
            self._error = str(e)
            print(f"[AUTO-RELOAD] Error: {e}")
            
            # Broadcast error event
            await self._broadcast_event("reload_error", {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        
        finally:
            self._reload_in_progress = False