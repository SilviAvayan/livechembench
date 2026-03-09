from src.services.chemrxiv_client import ChemRxivClient
from src.core.interfaces import PaperProvider

class ProviderFactory:
    """The Factory Pattern implementation."""
    @staticmethod
    def get_provider(provider_type: str) -> PaperProvider:
        providers = {
            "chemrxiv": ChemRxivClient,
            # "pubmed": PubMedClient, <--- Add more later!
        }
        
        target_class = providers.get(provider_type.lower())
        if not target_class:
            raise ValueError(f"Provider '{provider_type}' is not supported.")
            
        return target_class()