# Agent-GGUF

Automatically generate Ollama Modelfiles from GGUF files with metadata extraction and HuggingFace model search.

## Features

- 🔍 **GGUF Metadata Extraction**: Automatically parse GGUF files to extract model architecture, context length, chat templates, and more
- 🌐 **HuggingFace Integration**: Search and retrieve model information from HuggingFace Hub
- 📝 **Smart Template Selection**: Automatically select appropriate chat templates based on model architecture (Llama, Mistral, Phi, Gemma)
- 🎨 **Rich CLI Output**: Beautiful command-line interface with colored progress indicators, tables, and panels
- ⚙️ **Customizable**: Override model names, skip HuggingFace search, and customize output paths

## Installation

Install from PyPI:

```bash
pip install agent-gguf
```

Or install from source:

```bash
git clone https://github.com/bhongong/agent-gguf.git
cd agent-gguf
pip install -e .
```

## Usage

### Basic Usage

Generate a Modelfile from a GGUF file:

```bash
agent-gguf model.gguf
```

This will:
1. Parse the GGUF file metadata
2. Search for the model on HuggingFace
3. Generate a `Modelfile` in the current directory

### Advanced Usage

**Specify output file:**
```bash
agent-gguf model.gguf -o MyModelfile
```

**Skip HuggingFace search:**
```bash
agent-gguf model.gguf --no-search
```

**Override model name:**
```bash
agent-gguf model.gguf --model-name "custom-model-name"
```

**Combine options:**
```bash
agent-gguf model.gguf -o custom.modelfile --no-search --model-name "MyModel"
```

## Python API

You can also use Agent-GGUF programmatically:

```python
from agent_gguf import GGUFParser, HuggingFaceSearcher, ModelfileGenerator

# Parse GGUF file
parser = GGUFParser("model.gguf")
metadata = parser.parse()

# Search HuggingFace (optional)
searcher = HuggingFaceSearcher()
hf_info = searcher.search(metadata["clean_model_name"])

# Generate Modelfile
generator = ModelfileGenerator()
modelfile = generator.generate("model.gguf", metadata, hf_info)

# Save to file
with open("Modelfile", "w") as f:
    f.write(modelfile)
```

## Generated Modelfile Structure

The generated Modelfile includes:

- **FROM**: Path to your GGUF file
- **TEMPLATE**: Chat template based on model architecture
- **PARAMETER**: Default parameters (temperature, top_p, top_k, repeat_penalty)
- **SYSTEM**: Default system message

Example output:
```dockerfile
FROM model.gguf

TEMPLATE """
{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """
You are Llama-3-8B, a helpful AI assistant.
"""
```

## Supported Architectures

Agent-GGUF includes built-in chat templates for:

- **Llama** (Llama 2, Llama 3, etc.)
- **Mistral** (Mistral 7B, Mixtral, etc.)
- **Phi** (Phi-2, Phi-3, etc.)
- **Gemma** (Gemma 2B, 7B, etc.)

For other architectures, it will attempt to use the chat template from the GGUF file metadata or default to the Llama template.

## Requirements

- Python 3.8+
- gguf>=0.1.0
- huggingface-hub>=0.16.0
- click>=8.0.0
- rich>=10.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Author

**Calixto Ong**  
Email: bhongong@gmail.com  
GitHub: [@bhongong](https://github.com/bhongong)

## Acknowledgments

- Built with [gguf](https://github.com/ggerganov/ggml) for GGUF file parsing
- Powered by [HuggingFace Hub](https://huggingface.co/) for model search
- Beautiful CLI thanks to [Rich](https://github.com/Textualize/rich)
