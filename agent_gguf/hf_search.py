"""HuggingFace model search functionality."""

from typing import Any, Dict, Optional

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


class HuggingFaceSearcher:
    """Search for models on HuggingFace."""
    
    def __init__(self):
        """Initialize the HuggingFace searcher."""
        if HfApi is None:
            raise ImportError(
                "huggingface_hub library not installed. "
                "Install with: pip install huggingface-hub"
            )
        self.api = HfApi()
    
    def search(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Search for a model on HuggingFace.
        
        Args:
            model_name: Name of the model to search for
            
        Returns:
            Dictionary containing model information or None if not found
        """
        # Clean the model name
        clean_name = self._clean_model_name(model_name)
        
        try:
            # First try: search with library="gguf" filter
            results = self._search_with_filter(clean_name, library="gguf")
            if results:
                return results
            
            # Second try: broader search without library filter
            results = self._search_with_filter(clean_name)
            if results:
                return results
            
            # Third try: search with original name
            if clean_name != model_name:
                results = self._search_with_filter(model_name, library="gguf")
                if results:
                    return results
                results = self._search_with_filter(model_name)
                if results:
                    return results
            
            return None
            
        except Exception as e:
            return {
                "error": str(e),
                "model_id": None,
            }
    
    def _search_with_filter(
        self, 
        query: str, 
        library: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Search with specific filters.
        
        Args:
            query: Search query
            library: Optional library filter
            
        Returns:
            Model information or None
        """
        try:
            kwargs = {
                "search": query,
                "sort": "downloads",
                "direction": -1,
                "limit": 5,
            }
            
            if library:
                kwargs["library"] = library
            
            models = list(self.api.list_models(**kwargs))
            
            if not models:
                return None
            
            # Get the most downloaded model
            model = models[0]
            
            # Extract model information
            return {
                "model_id": model.id,
                "author": model.author if hasattr(model, "author") else model.id.split("/")[0] if "/" in model.id else None,
                "downloads": model.downloads if hasattr(model, "downloads") else 0,
                "tags": model.tags if hasattr(model, "tags") else [],
                "base_model": self._extract_base_model(model),
                "license": model.card_data.get("license") if hasattr(model, "card_data") and model.card_data else None,
            }
            
        except Exception:
            return None
    
    def _clean_model_name(self, name: str) -> str:
        """Clean model name by removing quantization suffixes.
        
        Args:
            name: Original model name
            
        Returns:
            Cleaned model name
        """
        import re
        
        # Remove quantization patterns
        patterns = [
            r'-[qQ]\d+_[kK]_[mMsSlL]',
            r'-[qQ]\d+_[kK]',
            r'-[qQ]\d+',
            r'\.gguf$',
        ]
        
        for pattern in patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        return name.strip('-_')
    
    def _extract_base_model(self, model: Any) -> Optional[str]:
        """Extract base model information.
        
        Args:
            model: HuggingFace model object
            
        Returns:
            Base model name or None
        """
        try:
            if hasattr(model, "card_data") and model.card_data:
                base_model = model.card_data.get("base_model")
                if base_model:
                    if isinstance(base_model, list):
                        return base_model[0] if base_model else None
                    return base_model
        except Exception:
            pass
        
        return None
