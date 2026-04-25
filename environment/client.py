import httpx
from typing import Optional, Dict, Any
from .track_a import RefactorAction, RefactorObservation

class ConstrainedRefactorClient:
    """OpenEnv Client for interacting with the Constrained Refactor Gauntlet."""
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        self.sync_client = httpx.Client(timeout=30.0)

    async def reset(self) -> RefactorObservation:
        response = await self.client.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return RefactorObservation(**response.json()["observation"])

    async def step(self, action: RefactorAction) -> Dict[str, Any]:
        response = await self.client.post(f"{self.base_url}/step", json=action.dict())
        response.raise_for_status()
        return response.json()

    def sync(self):
        class SyncClient:
            def __init__(self, client):
                self.client = client
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
            def reset(self) -> RefactorObservation:
                response = self.client.sync_client.post(f"{self.client.base_url}/reset")
                response.raise_for_status()
                return RefactorObservation(**response.json()["observation"])
                
            def step(self, action: RefactorAction) -> Dict[str, Any]:
                response = self.client.sync_client.post(f"{self.client.base_url}/step", json=action.dict())
                response.raise_for_status()
                return response.json()
                
        return SyncClient(self)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        self.sync_client.close()
