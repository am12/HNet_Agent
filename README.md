<p align="center">
  <h1 align="center">HNet_Agent</h1>
</p>

# HNet_Agent: Benchmarking AI Agents on Hierarchical Language Models

## 📖 Overview
`HNet_Agent` is a benchmark and evaluation framework for testing AI agents on notebook-based tasks, built around **H-Net** — a state-of-the-art hierarchical language model with dynamic chunking for end-to-end sequence modeling.

**Key Components:**
- **H-Net Model**: Implementation of [H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955) (Hwang et al., 2025)
- **Benchmark Tools**: Extract questions from executed notebooks, run agent evaluations, and assess responses using LLM judges
- **MCP Server**: Model Context Protocol integration for Claude Code

## 🚀 Quick Start

### Basic Benchmark Evaluation
Run an AI agent evaluation on extracted benchmark questions:

> **⚠️ Prerequisites**: Complete the [installation & setup](#️-installation--setup) below before running evaluations.

```bash
cd HNet_Agent

python tools/benchmark_assessor.py \
  --input benchmark_questions.csv \
  --output results.csv \
  --judge-agent templates/judge-agent.md \
  --agent-def templates/solver-agent.md
```

### Extract Questions from Notebooks
Extract benchmark questions from an executed Jupyter notebook:

```bash
python tools/benchmark_extractor.py \
  <notebook.ipynb> \
  <output.csv>
```

### H-Net Text Generation
Generate text using pretrained H-Net models:

```bash
python repo/hnet/generate.py \
  --model-path notebooks/h_net_text_generation_script/data/hnet_2stage_L.pt \
  --config-path notebooks/h_net_text_generation_script/data/hnet_2stage_L.json \
  --max-tokens 1024 \
  --temperature 1.0 \
  --top-p 1.0
```

### Parameters

**Benchmark Assessor:**
- `--input <file>`: CSV file with benchmark questions
- `--output <file>`: Output CSV for results
- `--judge-agent <file>`: Agent definition for LLM judge
- `--agent-def <file>`: Agent definition for solver

**Benchmark Extractor:**
- `<notebook>`: Input Jupyter notebook (executed)
- `<output>`: Output CSV file for questions

**Text Generation:**
- `--model-path <file>`: Path to pretrained H-Net model weights
- `--config-path <file>`: Path to model configuration JSON
- `--max-tokens <int>`: Maximum tokens to generate
- `--temperature <float>`: Sampling temperature
- `--top-p <float>`: Nucleus sampling parameter

## ⚙️ Installation & Setup

### Prerequisites
- **Python**: Version 3.10 or higher
- **CUDA** (optional): Required for GPU acceleration with Triton, Flash Attention, and Mamba SSM
- **Claude Code** (optional): For MCP server integration

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd HNet_Agent
   ```

2. **Create Virtual Environment**
   ```bash
   python3.10 -m venv hnet-env
   source hnet-env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install H-Net in Development Mode**
   ```bash
   cd repo/hnet
   pip install -e .
   cd ../..
   ```

5. **(Optional) Install GPU Dependencies**
   > **Note**: GPU-specific packages (triton, mamba_ssm, causal_conv1d, flash_attn) require CUDA and are incompatible with macOS ARM.

   ```bash
   pip install triton>=3.2.0
   pip install flash_attn==2.8.0.post2
   pip install mamba_ssm causal_conv1d
   ```

6. **(Optional) Install and Configure Claude Code**
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude
   ```

## 🤖 MCP Server Integration

### Connect HNet_Agent to Claude Code
To use the HNet_Agent tools as an MCP server in Claude Code:

```bash
fastmcp install claude-code src/HNet_Agent_mcp.py \
  --python hnet-env/bin/python
```

### Verification
Verify your MCP server is loaded:
```bash
claude mcp list
```
or use `/mcp` inside Claude Code. You should see `HNet_Agent` listed.

## 📁 Project Structure

```
HNet_Agent/
├── repo/hnet/                      # H-Net Language Model Implementation
│   ├── hnet/
│   │   ├── models/                 # Model implementations
│   │   │   ├── hnet.py             # Main H-Net hierarchical model
│   │   │   ├── mixer_seq.py        # HNetForCausalLM wrapper
│   │   │   └── config_hnet.py      # Configuration dataclasses
│   │   ├── modules/                # Model components
│   │   │   ├── dc.py               # Dynamic Chunking modules
│   │   │   ├── isotropic.py        # Non-hierarchical components
│   │   │   ├── block.py            # Mamba2/Attention blocks
│   │   │   ├── mha.py              # Multi-Head Attention
│   │   │   └── rotary.py           # Rotary positional embeddings
│   │   └── utils/                  # Training and tokenization utilities
│   ├── generate.py                 # Text generation script
│   └── pyproject.toml              # Package metadata
├── src/
│   └── HNet_Agent_mcp.py           # MCP server for HNet_Agent tools
├── tools/                          # Benchmark Evaluation Tools
│   ├── benchmark_assessor.py       # Run benchmark assessments via Claude CLI
│   ├── benchmark_extractor.py      # Extract questions from notebooks
│   ├── benchmark_reviewer.py       # Review and filter benchmark questions
│   ├── preprocess_notebook.py      # Preprocess notebooks for LLM
│   └── extract_notebook_images.py  # Extract images from notebooks
├── notebooks/                      # Tutorial Notebooks
│   └── h_net_text_generation_script/
│       ├── *_execution.ipynb       # Executed tutorial notebooks
│       └── data/                   # Pretrained model weights
├── tests/                          # Test suite
├── templates/                      # Agent definition templates
├── requirements.txt                # Python dependencies
└── pytest.ini                      # Pytest configuration
```

### Key Files

| File/Directory | Description |
|----------------|-------------|
| `src/HNet_Agent_mcp.py` | MCP server exposing benchmark tools to Claude Code |
| `tools/benchmark_assessor.py` | Main benchmark evaluation driver |
| `tools/benchmark_extractor.py` | Extract questions from executed notebooks |
| `repo/hnet/generate.py` | H-Net text generation CLI |
| `repo/hnet/hnet/models/hnet.py` | Core H-Net hierarchical architecture |
| `notebooks/*/data/*.pt` | Pretrained H-Net model weights |

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest -m unit

# Run with verbose output
pytest -v

# Skip slow tests
pytest -m "not slow"
```

## 🔬 H-Net Architecture

H-Net introduces a hierarchical sequence modeling approach with dynamic chunking:

- **Multi-Stage Hierarchy**: Recursive encoder-decoder structure at each level
- **Dynamic Chunking**: RoutingModule learns to segment sequences adaptively
- **Hybrid Blocks**: Combines Mamba2 SSM and Causal Multi-Head Attention
- **ByteTokenizer**: Byte-level tokenization (vocab_size=256)

**Pretrained Models Available:**
- `hnet_1stage_L` / `hnet_2stage_L` — English text
- `hnet_1stage_XL` / `hnet_2stage_XL` — Larger English models
- Chinese and Code variants

## 📊 Benchmark Workflow

1. **Execute Tutorial Notebooks**: Run Jupyter notebooks with `papermill`
2. **Extract Questions**: Use `benchmark_extractor.py` to generate questions from outputs
3. **Review Questions**: Filter and refine with `benchmark_reviewer.py`
4. **Run Assessment**: Evaluate AI agents with `benchmark_assessor.py`
5. **Analyze Results**: Review CSV output with scores and explanations

## 📚 Citation

```bibtex
@article{hwang2025hnet,
  title={H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
  author={Hwang, Sukjun and Wang, Brandon and Gu, Albert},
  journal={arXiv preprint arXiv:2507.07955},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License. See `repo/hnet/LICENSE` for details.
