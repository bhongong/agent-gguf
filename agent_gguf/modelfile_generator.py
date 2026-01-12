"""Modelfile generator for Ollama."""

from typing import Dict, Optional


class ModelfileGenerator:
    """Generate Ollama Modelfiles from GGUF metadata."""
    
    # Template mappings using proper Go template syntax
    TEMPLATE_MAPPINGS = {
        "llama": """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>""",
        "mistral": """{{ if .System }}[INST] {{ .System }} [/INST]{{ end }}{{ if .Prompt }}[INST] {{ .Prompt }} [/INST]{{ end }}{{ .Response }}""",
        "phi": """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
""",
        "gemma": """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
""",
    }
    
    # Default parameters
    DEFAULT_PARAMS = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
    }
    
    def __init__(self):
        """Initialize the Modelfile generator."""
        pass
    
    def generate(
        self,
        gguf_file: str,
        metadata: Dict[str, any],
        hf_info: Optional[Dict[str, any]] = None,
        custom_params: Optional[Dict[str, any]] = None,
    ) -> str:
        """Generate a complete Modelfile.
        
        Args:
            gguf_file: Path to the GGUF file
            metadata: Metadata extracted from GGUF file
            hf_info: Optional HuggingFace model information
            custom_params: Optional custom parameters
            
        Returns:
            Generated Modelfile content
        """
        lines = []
        
        # FROM directive
        lines.append(f"FROM {gguf_file}")
        lines.append("")
        
        # TEMPLATE directive
        template = self._select_template(metadata, hf_info)
        if template:
            lines.append("TEMPLATE \"\"\"")
            lines.append(template)
            lines.append('"""')
            lines.append("")
        
        # PARAMETER directives
        params = self.DEFAULT_PARAMS.copy()
        if custom_params:
            params.update(custom_params)
        
        for key, value in params.items():
            lines.append(f"PARAMETER {key} {value}")
        lines.append("")
        
        # SYSTEM directive
        system_message = self._generate_system_message(metadata, hf_info)
        if system_message:
            lines.append('SYSTEM """')
            lines.append(system_message)
            lines.append('"""')
        
        return "\n".join(lines)
    
    def _select_template(
        self,
        metadata: Dict[str, any],
        hf_info: Optional[Dict[str, any]] = None,
    ) -> Optional[str]:
        """Select the appropriate template based on architecture.
        
        Args:
            metadata: GGUF metadata
            hf_info: HuggingFace model information
            
        Returns:
            Template string or None
        """
        # First, try to use chat_template from metadata
        if metadata.get("chat_template"):
            return metadata["chat_template"]
        
        # Otherwise, select based on architecture
        architecture = metadata.get("architecture", "").lower()
        
        # Try exact match first
        if architecture in self.TEMPLATE_MAPPINGS:
            return self.TEMPLATE_MAPPINGS[architecture]
        
        # Try partial match
        for arch_name, template in self.TEMPLATE_MAPPINGS.items():
            if arch_name in architecture:
                return template
        
        # Check model name for architecture hints
        model_name = metadata.get("model_name", "").lower()
        for arch_name, template in self.TEMPLATE_MAPPINGS.items():
            if arch_name in model_name:
                return template
        
        # Check HuggingFace info
        if hf_info and hf_info.get("model_id"):
            model_id = hf_info["model_id"].lower()
            for arch_name, template in self.TEMPLATE_MAPPINGS.items():
                if arch_name in model_id:
                    return template
        
        # Default to llama template
        return self.TEMPLATE_MAPPINGS["llama"]
    
    def _generate_system_message(
        self,
        metadata: Dict[str, any],
        hf_info: Optional[Dict[str, any]] = None,
    ) -> str:
        """Generate a default system message.
        
        Args:
            metadata: GGUF metadata
            hf_info: HuggingFace model information
            
        Returns:
            System message
        """
        model_name = metadata.get("model_name", "Assistant")
        
        if hf_info and hf_info.get("model_id"):
            model_name = hf_info["model_id"]
        
        return f"You are {model_name}, a helpful AI assistant."
