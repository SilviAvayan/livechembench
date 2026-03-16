"""Factory for constructing concrete paper providers."""

from src.services.chemrxiv_client import ChemRxivClient
from src.services.pubmed_client import PubMedClient
from src.core.interfaces import PaperProvider


class ProviderFactory:
    """Simple factory for `PaperProvider` implementations."""

    _providers = {
        "chemrxiv": ChemRxivClient,
        "pubmed": PubMedClient,
    }

    @staticmethod
    def get_provider(provider_type: str) -> PaperProvider:
        """Instantiate a provider by name (e.g. 'chemrxiv', 'pubmed')."""
        key = provider_type.lower()
        target_class = ProviderFactory._providers.get(key)
        if not target_class:
            raise ValueError(f"Provider '{provider_type}' is not supported.")

        return target_class()