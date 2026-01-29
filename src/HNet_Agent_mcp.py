"""
Model Context Protocol (MCP) for HNet_Agent

HNet_Agent provides tools for benchmarking and evaluating AI agents on notebook-based tasks.
It includes utilities for extracting questions from tutorials, preprocessing notebooks, and running automated assessments.

NOTE: The current tools are CLI-based utilities that require external dependencies and subprocess execution.
This MCP server provides access to the core functionality but may require additional setup for full functionality.

This MCP Server contains documentation for the following utility files (tools require refactoring for full MCP integration):
1. benchmark_assessor.py
    - Command-line tool for running benchmark assessments using Claude CLI
    - Evaluates agent responses against ground truth using LLM judges
2. benchmark_extractor.py
    - Extracts and validates benchmark questions from executed notebooks
    - Filters out plotting-related questions and validates against cell outputs
3. benchmark_reviewer.py
    - Reviews and filters benchmark questions for quality and non-redundancy
    - Selects top 10-15 high-quality questions from candidate sets
4. extract_notebook_images.py
    - Extracts all images (PNG, JPEG, SVG) from Jupyter notebooks
    - Saves images with systematic naming conventions
5. preprocess_notebook.py
    - Preprocesses notebooks by removing images and truncating long text outputs
    - Prepares notebooks for LLM consumption while preserving structure
"""

from fastmcp import FastMCP
import subprocess
import json
import sys
from typing import Annotated
from pathlib import Path
import os

# Project root for relative paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# MCP server instance
mcp = FastMCP(name="HNet_Agent")

@mcp.tool
def run_benchmark_assessor(
    input_csv: Annotated[str, "Path to benchmark_questions.csv file"],
    output_csv: Annotated[str, "Path to save assessment results CSV"],
    judge_agent_md: Annotated[str, "Path to benchmark-judge.md definition file"],
    agent_def_md: Annotated[str, "Path to benchmark-solver.md definition file"]
) -> dict:
    """
    Run benchmark assessment using the benchmark_assessor CLI tool.
    Evaluates AI agent responses against ground truth using LLM judges.
    """
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "benchmark_assessor.py"),
            "--input", input_csv,
            "--output", output_csv,
            "--judge-agent", judge_agent_md,
            "--agent-def", agent_def_md
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode == 0:
            return {
                "success": True,
                "message": f"Benchmark assessment completed successfully",
                "output_file": output_csv,
                "stdout": result.stdout
            }
        else:
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to run benchmark assessor: {str(e)}"
        }

@mcp.tool
def extract_notebook_images(
    notebook_path: Annotated[str, "Path to the Jupyter notebook (.ipynb file)"],
    output_dir: Annotated[str, "Directory to save extracted images"]
) -> dict:
    """
    Extract all images (PNG, JPEG, SVG) from a Jupyter notebook.
    Images are saved with systematic naming: cell_X_output_Y_fig_Z.ext
    """
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "extract_notebook_images.py"),
            notebook_path,
            output_dir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode == 0:
            # Parse output to count extracted images
            lines = result.stdout.strip().split('\n')
            saved_files = [line.replace("Saved: ", "") for line in lines if line.startswith("Saved: ")]
            total_line = [line for line in lines if "Total images extracted:" in line]
            total_count = int(total_line[0].split(": ")[1]) if total_line else len(saved_files)

            return {
                "success": True,
                "message": f"Extracted {total_count} images from notebook",
                "output_directory": output_dir,
                "extracted_files": saved_files,
                "total_count": total_count
            }
        else:
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to extract images: {str(e)}"
        }

@mcp.tool
def preprocess_notebook(
    input_notebook: Annotated[str, "Path to input Jupyter notebook (.ipynb)"],
    output_notebook: Annotated[str, "Path for preprocessed output notebook"],
    max_text_length: Annotated[int, "Maximum characters for text outputs"] = 2000
) -> dict:
    """
    Preprocess a Jupyter notebook by removing images and truncating long text outputs.
    Prepares notebooks for LLM consumption while preserving code and structure.
    """
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "preprocess_notebook.py"),
            input_notebook,
            output_notebook,
            "--max_len", str(max_text_length)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode == 0:
            return {
                "success": True,
                "message": "Notebook preprocessing completed successfully",
                "input_file": input_notebook,
                "output_file": output_notebook,
                "max_text_length": max_text_length,
                "stdout": result.stdout
            }
        else:
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to preprocess notebook: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run()