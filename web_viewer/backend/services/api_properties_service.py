"""
API Properties Service

Fetches node properties from external REST APIs.
Designed to be extensible for adding new API providers.

Each provider must implement:
- fetch_all(): Fetch all data from the API (handles pagination)
- transform_to_df(): Convert API response to DataFrame with 'avatar' column
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import settings


class ExternalPropertyProvider(ABC):
    """Abstract base class for external API property providers."""
    
    # Provider identifier (used for caching and config)
    name: str = "base"
    
    # Human-readable name
    display_name: str = "Base Provider"
    
    def __init__(self):
        """Initialize provider with configured HTTP session."""
        self.session = self._create_session()
        self.base_url = settings.EXTERNAL_API_BASE_URL
        self.timeout = settings.EXTERNAL_API_TIMEOUT
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=settings.EXTERNAL_API_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @property
    @abstractmethod
    def endpoint(self) -> str:
        """API endpoint path (without base URL)."""
        pass
    
    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this provider is enabled via config."""
        pass
    
    @abstractmethod
    def get_api_params(self, version: str) -> Dict[str, Any]:
        """Get API query parameters for the given version."""
        pass
    
    @abstractmethod
    def transform_to_df(self, response_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform API response to DataFrame.
        
        Must return DataFrame with:
        - 'avatar' column (lowercase addresses)
        - One or more property columns
        """
        pass
    
    @property
    def columns_provided(self) -> List[str]:
        """List of column names this provider adds (excluding 'avatar')."""
        return []
    
    def fetch_all(self, version: str) -> pd.DataFrame:
        """
        Fetch all data from the API, handling pagination.
        
        Args:
            version: Graph version (e.g., 'v1', 'v2')
            
        Returns:
            DataFrame with all properties
        """
        if not self.enabled:
            print(f"[API-PROPS] Provider '{self.name}' is disabled")
            return pd.DataFrame()
        
        url = f"{self.base_url}{self.endpoint}"
        params = self.get_api_params(version)
        
        print(f"[API-PROPS] Fetching from {self.name}: {url}")
        start_time = time.time()
        
        all_data = []
        offset = 0
        page_size = params.get('limit', 1000)
        total = None
        
        try:
            while True:
                # Update offset for pagination
                current_params = {**params, 'offset': offset}
                
                response = self.session.get(
                    url,
                    params=current_params,
                    timeout=self.timeout,
                    headers={'Accept': 'application/json'}
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Check for error in response
                if data.get('status') == 'error':
                    print(f"[API-PROPS] API error: {data.get('message', 'Unknown error')}")
                    break
                
                # Get total count on first request
                if total is None:
                    total = data.get('total', 0)
                    print(f"[API-PROPS] Total records: {total}")
                
                # Get current batch
                batch = self._extract_items(data)
                if not batch:
                    break
                
                all_data.extend(batch)
                
                # Check if we have all data
                count = data.get('count', len(batch))
                offset += count
                
                if offset >= total:
                    break
                
                # Small delay to be nice to the API
                time.sleep(0.1)
            
            elapsed = time.time() - start_time
            print(f"[API-PROPS] Fetched {len(all_data)} records from {self.name} in {elapsed:.2f}s")
            
            # Transform to DataFrame
            if all_data:
                df = self.transform_to_df({'addresses': all_data})
                return df
            
            return pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            print(f"[API-PROPS] Error fetching from {self.name}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"[API-PROPS] Unexpected error in {self.name}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _extract_items(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract items list from API response. Override if needed."""
        return response_data.get('addresses', [])


class BlacklistProvider(ExternalPropertyProvider):
    """Provider for blacklist/bot detection data."""
    
    name = "blacklist"
    display_name = "Blacklist (Bot Detection)"
    
    @property
    def endpoint(self) -> str:
        return settings.EXTERNAL_API_BLACKLIST_ENDPOINT
    
    @property
    def enabled(self) -> bool:
        return settings.EXTERNAL_API_BLACKLIST_ENABLED
    
    @property
    def columns_provided(self) -> List[str]:
        return ['isBlacklisted', 'blacklistReason']
    
    def get_api_params(self, version: str) -> Dict[str, Any]:
        """Get API parameters based on version."""
        params = {
            'include_reason': 'true',
            'limit': 1000,
        }
        
        # Apply v2_only filter if configured and version is v2
        if settings.EXTERNAL_API_BLACKLIST_V2_ONLY and version == 'v2':
            params['v2_only'] = 'true'
        
        return params
    
    def transform_to_df(self, response_data: Dict[str, Any]) -> pd.DataFrame:
        """Transform blacklist API response to DataFrame."""
        addresses = response_data.get('addresses', [])
        
        if not addresses:
            return pd.DataFrame(columns=['avatar', 'isBlacklisted', 'blacklistReason'])
        
        rows = []
        for item in addresses:
            address = item.get('address', '')
            if address:
                rows.append({
                    'avatar': address.lower(),
                    'isBlacklisted': True,
                    'blacklistReason': item.get('reason', 'unknown')
                })
        
        df = pd.DataFrame(rows)
        
        # Ensure correct types
        if not df.empty:
            df['avatar'] = df['avatar'].astype(str)
            df['isBlacklisted'] = df['isBlacklisted'].astype(bool)
            df['blacklistReason'] = df['blacklistReason'].astype(str)
        
        return df


# =============================================================================
# Provider Registry
# =============================================================================

# Register all available providers here
PROVIDER_CLASSES: Dict[str, Type[ExternalPropertyProvider]] = {
    'blacklist': BlacklistProvider,
    # Add new providers here:
    # 'reputation': ReputationProvider,
    # 'labels': LabelsProvider,
}


class APIPropertiesService:
    """
    Service for managing external API property providers.
    
    Coordinates fetching from multiple providers and merging results.
    """
    
    def __init__(self):
        """Initialize service with configured providers."""
        self._providers: Dict[str, ExternalPropertyProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize enabled providers."""
        enabled_providers = settings.EXTERNAL_API_PROVIDERS
        
        for provider_name in enabled_providers:
            if provider_name in PROVIDER_CLASSES:
                provider = PROVIDER_CLASSES[provider_name]()
                if provider.enabled:
                    self._providers[provider_name] = provider
                    print(f"[API-PROPS] Registered provider: {provider_name}")
                else:
                    print(f"[API-PROPS] Provider disabled: {provider_name}")
            else:
                print(f"[API-PROPS] Unknown provider: {provider_name}")
    
    @property
    def available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available (enabled) providers with metadata."""
        return [
            {
                'name': p.name,
                'display_name': p.display_name,
                'columns': p.columns_provided,
                'enabled': p.enabled
            }
            for p in self._providers.values()
        ]
    
    @property
    def all_columns_provided(self) -> List[str]:
        """Get all column names provided by all enabled providers."""
        columns = []
        for provider in self._providers.values():
            columns.extend(provider.columns_provided)
        return columns
    
    def fetch_all_providers(
        self,
        version: str,
        providers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch properties from all (or specified) providers.
        
        Args:
            version: Graph version (e.g., 'v1', 'v2')
            providers: Optional list of provider names to use (None = all)
            
        Returns:
            Merged DataFrame with 'avatar' and all property columns
        """
        target_providers = self._providers
        
        if providers:
            target_providers = {
                k: v for k, v in self._providers.items() 
                if k in providers
            }
        
        if not target_providers:
            print("[API-PROPS] No providers to fetch from")
            return pd.DataFrame()
        
        start_time = time.time()
        all_dfs: List[pd.DataFrame] = []
        columns_loaded: List[str] = []
        
        for name, provider in target_providers.items():
            try:
                df = provider.fetch_all(version)
                if not df.empty:
                    all_dfs.append(df)
                    columns_loaded.extend(provider.columns_provided)
            except Exception as e:
                print(f"[API-PROPS] Failed to fetch from {name}: {e}")
                continue
        
        if not all_dfs:
            print("[API-PROPS] No data fetched from any provider")
            return pd.DataFrame()
        
        # Merge all DataFrames on 'avatar'
        result_df = all_dfs[0]
        for df in all_dfs[1:]:
            result_df = result_df.merge(df, on='avatar', how='outer')
        
        elapsed = time.time() - start_time
        print(f"[API-PROPS] Total: {len(result_df)} rows, "
              f"{len(columns_loaded)} columns in {elapsed:.2f}s")
        
        return result_df
    
    def get_provider(self, name: str) -> Optional[ExternalPropertyProvider]:
        """Get a specific provider by name."""
        return self._providers.get(name)


# Singleton instance
api_properties_service = APIPropertiesService()