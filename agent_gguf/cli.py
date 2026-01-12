"""Command-line interface for Agent-GGUF."""

import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    click = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
except ImportError:
    Console = None
    Table = None
    Panel = None
    Progress = None
    Syntax = None

from .gguf_parser import GGUFParser
from .hf_search import HuggingFaceSearcher
from .modelfile_generator import ModelfileGenerator


class CLI:
    """Command-line interface for Agent-GGUF."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.console = Console() if Console else None
    
    def print(self, message: str, style: Optional[str] = None):
        """Print a message with optional styling.
        
        Args:
            message: Message to print
            style: Optional style (for rich console)
        """
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def print_panel(self, message: str, title: str, style: str = "green"):
        """Print a message in a panel.
        
        Args:
            message: Message to print
            title: Panel title
            style: Panel style
        """
        if self.console and Panel:
            self.console.print(Panel(message, title=title, style=style))
        else:
            print(f"\n=== {title} ===")
            print(message)
            print("=" * (len(title) + 8))
    
    def print_table(self, title: str, data: dict):
        """Print data in a table format.
        
        Args:
            title: Table title
            data: Dictionary of key-value pairs
        """
        if self.console and Table:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            
            for key, value in data.items():
                if value is not None:
                    table.add_row(key, str(value))
            
            self.console.print(table)
        else:
            print(f"\n{title}")
            print("-" * 50)
            for key, value in data.items():
                if value is not None:
                    print(f"{key}: {value}")
            print()
    
    def run(
        self,
        gguf_file: str,
        output: Optional[str] = None,
        no_search: bool = False,
        model_name: Optional[str] = None,
    ):
        """Run the main CLI workflow.
        
        Args:
            gguf_file: Path to GGUF file
            output: Optional output file path
            no_search: Skip HuggingFace search
            model_name: Optional custom model name
        """
        try:
            # Step 1: Parse GGUF file
            self.print("\n[bold blue]Step 1: Parsing GGUF file...[/bold blue]")
            parser = GGUFParser(gguf_file)
            metadata = parser.parse()
            
            if model_name:
                metadata["model_name"] = model_name
                metadata["clean_model_name"] = model_name
            
            self.print_table("GGUF Metadata", {
                "File": gguf_file,
                "Model Name": metadata.get("model_name"),
                "Architecture": metadata.get("architecture"),
                "Context Length": metadata.get("context_length"),
                "Quantization": metadata.get("quantization"),
            })
            
            # Step 2: Search HuggingFace (if not disabled)
            hf_info = None
            if not no_search:
                self.print("\n[bold blue]Step 2: Searching HuggingFace...[/bold blue]")
                try:
                    searcher = HuggingFaceSearcher()
                    search_name = metadata.get("clean_model_name") or metadata.get("model_name")
                    hf_info = searcher.search(search_name)
                    
                    if hf_info and not hf_info.get("error"):
                        self.print_table("HuggingFace Model Info", {
                            "Model ID": hf_info.get("model_id"),
                            "Author": hf_info.get("author"),
                            "Downloads": hf_info.get("downloads"),
                            "License": hf_info.get("license"),
                            "Base Model": hf_info.get("base_model"),
                        })
                    else:
                        self.print("[yellow]No matching model found on HuggingFace[/yellow]")
                except Exception as e:
                    self.print(f"[yellow]HuggingFace search failed: {e}[/yellow]")
            else:
                self.print("\n[bold blue]Step 2: Skipping HuggingFace search[/bold blue]")
            
            # Step 3: Generate Modelfile
            self.print("\n[bold blue]Step 3: Generating Modelfile...[/bold blue]")
            generator = ModelfileGenerator()
            modelfile_content = generator.generate(gguf_file, metadata, hf_info)
            
            # Determine output path
            if output:
                output_path = Path(output)
            else:
                output_path = Path("Modelfile")
            
            # Write to file
            output_path.write_text(modelfile_content)
            
            self.print_panel(
                f"Modelfile successfully generated!\n\nOutput: {output_path.absolute()}",
                "Success",
                "green",
            )
            
            # Show preview
            self.print("\n[bold]Modelfile Preview:[/bold]")
            if self.console and Syntax:
                syntax = Syntax(modelfile_content, "dockerfile", theme="monokai", line_numbers=True)
                self.console.print(syntax)
            else:
                print(modelfile_content)
            
        except FileNotFoundError as e:
            self.print_panel(str(e), "Error", "red")
            sys.exit(1)
        except ImportError as e:
            self.print_panel(
                f"{e}\n\nPlease install required dependencies:\n"
                "pip install agent-gguf",
                "Error",
                "red",
            )
            sys.exit(1)
        except Exception as e:
            self.print_panel(f"Unexpected error: {e}", "Error", "red")
            sys.exit(1)


def main():
    """Main entry point for the CLI."""
    if click is None:
        print("Error: click library not installed. Install with: pip install click")
        sys.exit(1)
    
    @click.command()
    @click.argument('gguf_file', type=click.Path(exists=True))
    @click.option(
        '--output', '-o',
        type=click.Path(),
        help='Output file path (default: Modelfile)',
    )
    @click.option(
        '--no-search',
        is_flag=True,
        help='Skip HuggingFace search',
    )
    @click.option(
        '--model-name',
        type=str,
        help='Custom model name (overrides auto-detection)',
    )
    def cli_command(gguf_file, output, no_search, model_name):
        """Generate Ollama Modelfile from GGUF file.
        
        GGUF_FILE is the path to the GGUF model file.
        """
        cli = CLI()
        cli.run(gguf_file, output, no_search, model_name)
    
    cli_command()


if __name__ == "__main__":
    main()
