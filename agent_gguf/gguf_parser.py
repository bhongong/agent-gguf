"""GGUF file parser for extracting metadata."""

import re
from pathlib import Path
from typing import Dict, Optional

try:
    import gguf
except ImportError:
    gguf = None


class GGUFParser:
    """Parser for GGUF files to extract metadata."""

    def __init__(self, gguf_file: str):
        """Initialize the parser with a GGUF file path.
        
        Args:
            gguf_file: Path to the GGUF file
        """
        self.gguf_file = Path(gguf_file)
        if not self.gguf_file.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_file}")
        
        self.metadata: Dict[str, any] = {}
        
    def parse(self) -> Dict[str, any]:
        """Parse the GGUF file and extract metadata.
        
        Returns:
            Dictionary containing extracted metadata
        """
        if gguf is None:
            raise ImportError(
                "gguf library not installed. Install with: pip install gguf"
            )
        
        try:
            reader = gguf.GGUFReader(str(self.gguf_file))
            
            # Extract metadata from GGUF fields
            self.metadata = {
                "architecture": self._get_field(reader, "general.architecture"),
                "context_length": self._get_field(
                    reader, 
                    ["general.context_length", "llama.context_length"]
                ),
                "chat_template": self._get_field(
                    reader, 
                    ["tokenizer.chat_template", "general.chat_template"]
                ),
                "model_name": self._get_field(reader, "general.name"),
                "quantization": self._get_field(
                    reader,
                    ["general.quantization_version", "general.file_type"]
                ),
            }
            
            # Infer model name from filename if not found in metadata
            if not self.metadata.get("model_name"):
                self.metadata["model_name"] = self._infer_model_name()
            
            # Clean up the model name
            self.metadata["clean_model_name"] = self._clean_model_name(
                self.metadata.get("model_name", "")
            )
            
        except Exception as e:
            # Fallback to filename-based inference
            self.metadata = {
                "architecture": None,
                "context_length": None,
                "chat_template": None,
                "model_name": self._infer_model_name(),
                "quantization": self._infer_quantization(),
                "clean_model_name": self._clean_model_name(self._infer_model_name()),
                "parse_error": str(e),
            }
        
        return self.metadata
    
    def _get_field(self, reader: any, field_names: any) -> Optional[any]:
        """Get a field value from the GGUF reader.
        
        Args:
            reader: GGUF reader instance
            field_names: Single field name or list of field names to try
            
        Returns:
            Field value or None if not found
        """
        if isinstance(field_names, str):
            field_names = [field_names]
        
        for field_name in field_names:
            try:
                for field in reader.fields.values():
                    if field.name == field_name:
                        # Handle different field types
                        if hasattr(field, 'parts'):
                            return field.parts[-1].tobytes().decode('utf-8', errors='ignore').rstrip('\x00')
                        elif hasattr(field, 'data'):
                            if isinstance(field.data, bytes):
                                return field.data.decode('utf-8', errors='ignore').rstrip('\x00')
                            return field.data
                        return str(field)
            except Exception:
                continue
        
        return None
    
    def _infer_model_name(self) -> str:
        """Infer model name from filename.
        
        Returns:
            Inferred model name
        """
        filename = self.gguf_file.stem
        # Remove common quantization suffixes
        return self._clean_model_name(filename)
    
    def _clean_model_name(self, name: str) -> str:
        """Clean model name by removing quantization suffixes.
        
        Args:
            name: Original model name
            
        Returns:
            Cleaned model name
        """
        # Remove quantization patterns like -q4_k_m, -Q5_K_S, etc.
        patterns = [
            r'-[qQ]\d+_[kK]_[mMsSlL]',  # -q4_k_m, -Q5_K_S, etc.
            r'-[qQ]\d+_[kK]',           # -q4_k, -Q5_K, etc.
            r'-[qQ]\d+',                # -q4, -Q5, etc.
            r'\.gguf$',                 # .gguf extension
        ]
        
        for pattern in patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        return name.strip('-_')
    
    def _infer_quantization(self) -> Optional[str]:
        """Infer quantization from filename.
        
        Returns:
            Inferred quantization or None
        """
        filename = self.gguf_file.stem
        # Extract quantization pattern
        match = re.search(r'[qQ]\d+_[kK]_[mMsSlL]|[qQ]\d+_[kK]|[qQ]\d+', filename)
        if match:
            return match.group(0).upper()
        return None
