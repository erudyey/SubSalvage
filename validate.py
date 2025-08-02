#!/usr/bin/env python3
"""Validation script for SubSalvage installation and dependencies."""

import platform
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements."""
    version = sys.version_info
    required = (3, 12)
    
    if version[:2] >= required:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.12)"


def check_torch_cuda() -> Tuple[bool, str]:
    """Check PyTorch and CUDA availability."""
    try:
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"PyTorch {torch_version}, CUDA {cuda_version}, {device_count} GPU(s): {device_name}"
        else:
            return False, f"PyTorch {torch_version}, CUDA not available"
            
    except Exception as e:
        return False, f"PyTorch check failed: {e}"


def check_easyocr() -> Tuple[bool, str]:
    """Check EasyOCR installation and GPU support."""
    try:
        import easyocr
        
        # Try to create a reader to test GPU support
        try:
            reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
            return True, f"EasyOCR available with GPU: {torch.cuda.is_available()}"
        except Exception as e:
            return False, f"EasyOCR initialization failed: {e}"
            
    except ImportError:
        return False, "EasyOCR not installed"


def check_opencv() -> Tuple[bool, str]:
    """Check OpenCV installation."""
    try:
        import cv2
        version = cv2.__version__
        return True, f"OpenCV {version}"
    except ImportError:
        return False, "OpenCV not installed"


def check_ffmpeg() -> Tuple[bool, str]:
    """Check FFmpeg availability."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            # Extract version from first line
            first_line = result.stdout.split('\n')[0]
            return True, first_line
        else:
            return False, "FFmpeg command failed"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg command timed out"
    except FileNotFoundError:
        return False, "FFmpeg not found in PATH"
    except Exception as e:
        return False, f"FFmpeg check failed: {e}"


def check_dependencies() -> Tuple[bool, str]:
    """Check for dependency conflicts."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pipdeptree', '--warn', 'conflict'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            if "Warning!!" in result.stdout:
                return False, "Dependency conflicts detected"
            else:
                return True, "No dependency conflicts"
        else:
            return False, f"pipdeptree failed: {result.stderr}"
            
    except Exception as e:
        return False, f"Dependency check failed: {e}"


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "Platform": platform.platform(),
        "Architecture": platform.architecture()[0],
        "Processor": platform.processor(),
    }


def main() -> None:
    """Run validation checks."""
    console.print("\n[bold cyan]🔍 SubSalvage Installation Validation[/bold cyan]")
    
    # System Information
    console.print("\n[bold blue]System Information:[/bold blue]")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        console.print(f"  {key}: {value}")
    
    # Validation checks
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", check_torch_cuda),
        ("EasyOCR", check_easyocr),
        ("OpenCV", check_opencv),
        ("FFmpeg", check_ffmpeg),
        ("Dependencies", check_dependencies),
    ]
    
    console.print("\n[bold blue]Validation Results:[/bold blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=15)
    table.add_column("Status", width=8)
    table.add_column("Details", style="dim")
    
    all_passed = True
    
    for name, check_func in checks:
        passed, details = check_func()
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        table.add_row(name, status, details)
        
        if not passed:
            all_passed = False
    
    console.print(table)
    
    # Summary
    if all_passed:
        summary = Panel(
            "[green]✅ All validation checks passed![/green]\n"
            "[dim]SubSalvage is ready to use.[/dim]",
            title="[bold green]Validation Summary[/bold green]",
            border_style="green"
        )
    else:
        summary = Panel(
            "[red]❌ Some validation checks failed.[/red]\n"
            "[dim]Please address the issues above before using SubSalvage.[/dim]",
            title="[bold red]Validation Summary[/bold red]",
            border_style="red"
        )
    
    console.print(f"\n{summary}")
    
    # Test GPU with PyTorch
    if torch.cuda.is_available():
        console.print(f"\n[bold green]GPU Test:[/bold green]")
        try:
            device = torch.device('cuda')
            test_tensor = torch.randn(10, 10).to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            console.print(f"  ✅ GPU computation successful: {result.device}")
        except Exception as e:
            console.print(f"  ❌ GPU test failed: {e}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()