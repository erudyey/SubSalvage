#!/usr/bin/env python3
"""SubSalvage CLI - Extract hardcoded subtitles from videos."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(
    name="SubSalvage",
    help="Extract hardcoded subtitles from videos with high accuracy",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def extract(
    video_path: Path = typer.Argument(
        ...,
        help="Path to the input video file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output SRT file path (default: video_name.srt)"
    ),
    start_time: float = typer.Option(
        0.0, "--start", "-s", help="Start time in seconds (default: 0.0)"
    ),
    end_time: float = typer.Option(
        -1.0, "--end", "-e", help="End time in seconds (default: full video)"
    ),
    gpu: bool = typer.Option(
        True, "--gpu/--no-gpu", help="Use GPU acceleration if available (default: True)"
    ),
) -> None:
    """Extract hardcoded subtitles from a video file and save as SRT."""

    # Set default output path
    if output is None:
        output = video_path.with_suffix(".srt")

    # Display extraction info
    info_text = Text()
    info_text.append("Video: ", style="bold blue")
    info_text.append(str(video_path), style="green")
    info_text.append("\nOutput: ", style="bold blue")
    info_text.append(str(output), style="green")
    info_text.append("\nTime Range: ", style="bold blue")
    if end_time > 0:
        info_text.append(f"{start_time:.1f}s - {end_time:.1f}s", style="yellow")
    else:
        info_text.append(f"{start_time:.1f}s - end", style="yellow")
    info_text.append("\nGPU: ", style="bold blue")
    info_text.append("Enabled" if gpu else "Disabled", style="green" if gpu else "red")

    panel = Panel(
        info_text,
        title="[bold cyan]SubSalvage - Subtitle Extraction[/bold cyan]",
        border_style="cyan",
    )
    console.print(panel)

    # TODO: Implement actual extraction logic
    console.print("[yellow]⚠️  Extraction functionality not yet implemented[/yellow]")
    console.print(
        "[blue]💡 This will be implemented with OCR and upscaling strategies[/blue]"
    )


@app.command()
def validate() -> None:
    """Validate system requirements and GPU availability."""
    console.print("[bold cyan]🔍 Validating SubSalvage installation...[/bold cyan]")

    # TODO: Import and run validation script
    console.print("[yellow]⚠️  Validation functionality not yet implemented[/yellow]")
    console.print("[blue]💡 This will check PyTorch, CUDA, and dependencies[/blue]")


if __name__ == "__main__":
    app()
