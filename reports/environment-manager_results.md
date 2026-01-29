# HNet Environment Setup Report

**Date:** 2026-01-29
**Environment Name:** hnet-env
**Working Directory:** /private/tmp/Paper2Agent/HNet_Agent
**Target Codebase:** repo/hnet/

---

## Executive Summary

Successfully created a reproducible Python virtual environment for the HNet research codebase with CPU-compatible dependencies. The environment is functional for code inspection, structure analysis, and non-GPU testing. GPU-specific dependencies (triton, mamba_ssm, causal_conv1d, flash_attn) could not be installed on macOS ARM architecture due to CUDA requirements.

---

## Environment Configuration

### Python Version Selection

**Decision Rationale:**
- **Project Requirements:** Python >= 3.9 (from pyproject.toml)
- **Selected Version:** Python 3.10.19
- **Selection Logic:** Project specifies minimum Python 3.9, which is below our baseline of 3.10. Selected Python 3.10.19 for stability and broad compatibility.

### Environment Details

```
Environment Path: /private/tmp/Paper2Agent/HNet_Agent/hnet-env
Python Version: 3.10.19
Total Packages: 153
Activation Command: source /private/tmp/Paper2Agent/HNet_Agent/hnet-env/bin/activate
```

---

## Installation Summary

### Installation Method: Local Installation (pip install -e .)

**Installation Priority Applied:**
1. **PyPI** (Priority 1) - Not applicable (package not published on PyPI)
2. **Git URL** (Priority 2) - Not used (local development preferred)
3. **Local Installation** (Priority 3) - ✓ Used for development setup

**Rationale:** The HNet package is not available on PyPI. Following the README instructions, installed locally using `pip install -e .` for editable development mode.

### Core Dependencies Installed

**Successfully Installed (CPU-Compatible):**
- torch >= 2.5.1 (installed: 2.10.0)
- einops (0.8.2)
- optree (0.18.0)
- regex (2026.1.15)
- omegaconf (2.3.0)
- setuptools, wheel, packaging, ninja (build dependencies)

**GPU Dependencies (NOT Installed - macOS Incompatible):**
- triton >= 3.2.0 - No macOS ARM wheels available
- mamba_ssm - Requires CUDA/nvcc (Linux only)
- causal_conv1d - Requires CUDA/nvcc (Linux only)
- flash_attn == 2.8.0.post2 - Requires CUDA/nvcc (Linux only)

### Test Infrastructure

**Testing & Development Tools:**
- pytest (9.0.2)
- pytest-asyncio (1.3.0)
- papermill (2.6.0) - Jupyter notebook execution
- nbclient (0.10.4) - Notebook client
- ipykernel (7.1.0) - IPython kernel
- imagehash (4.3.2) - Image hashing utilities
- matplotlib (3.10.8) - Plotting library
- fastmcp (2.14.4) - MCP framework

---

## Test Configuration

### pytest.ini

Created standardized pytest configuration at:
```
/private/tmp/Paper2Agent/HNet_Agent/pytest.ini
```

**Configuration Highlights:**
- Test paths: `tests/`
- Test file patterns: `*_test.py`, `test_*.py`
- Verbose output with short tracebacks
- Markers for test categorization (unit, integration, slow)
- Deprecation warnings filtered

### conftest.py

Created global pytest configuration at:
```
/private/tmp/Paper2Agent/HNet_Agent/conftest.py
```

**Features:**
- Automatic project root path setup
- Matplotlib non-interactive backend for tests
- Auto-disable plt.show() during test execution

---

## Environment Validation Results

### Success Criteria Checklist

#### Environment Creation Validation
- [✓] **Python Version**: Python 3.10.19 selected based on project requirements (>= 3.9)
- [✓] **Clean Environment**: Fresh environment created as `hnet-env/` in working directory
- [✓] **Environment Activation**: Environment activates successfully

#### Dependency Installation Validation
- [✓] **Dependencies Installed**: Core CPU-compatible dependencies installed successfully
- [✓] **PyPI Priority**: Checked for PyPI availability (not published)
- [✗] **Import Verification**: Package import fails due to GPU dependency requirements in code
- [✓] **Custom Instructions**: Followed README installation instructions

#### Test Infrastructure Validation
- [✓] **Test Infrastructure**: pytest and supporting packages installed
- [✓] **Notebook Support**: papermill, nbclient, ipykernel installed
- [✓] **Test Files Created**: pytest.ini and conftest.py created in root
- [✓] **Configuration Integrity**: Pytest configuration loads without errors

#### Reproducibility Validation
- [✓] **Reproducibility**: Generated requirements.txt with 153 packages
- [✓] **Installation Documentation**: Installation method and limitations documented
- [✓] **Environment Summary**: Complete summary provided

---

## Known Limitations

### GPU Dependencies (macOS ARM)

The HNet codebase has hard dependencies on GPU-accelerated libraries that are only compatible with Linux + NVIDIA CUDA:

1. **triton >= 3.2.0**
   - Issue: No macOS ARM wheels available
   - Platform: Linux x86_64/aarch64 only

2. **mamba_ssm**
   - Issue: Requires nvcc compiler (CUDA toolkit)
   - Error: `bare_metal_version` not defined (missing CUDA)

3. **causal_conv1d**
   - Issue: Requires nvcc compiler (CUDA toolkit)
   - Error: `bare_metal_version` not defined (missing CUDA)

4. **flash_attn == 2.8.0.post2**
   - Issue: Requires CUDA_HOME environment variable
   - Error: CUDA install root not found

### Import Restrictions

**Module Import Status:**
- ✗ `import hnet` - Fails due to flash_attn import in hnet/modules/block.py
- ✓ Direct file access - Source code accessible at repo/hnet/
- ✓ Core dependencies - torch, einops, optree, regex, omegaconf work
- ✓ Test infrastructure - pytest, papermill, nbclient functional

**Code Structure Accessible:**
```
repo/hnet/hnet/
├── models/      # HNet model implementations
├── modules/     # Model components (DC, isotropic)
└── utils/       # Utility functions
```

---

## Reproducibility Instructions

### Environment Recreation

To recreate this environment on a compatible system:

```bash
# 1. Install uv package manager
pip install uv

# 2. Create virtual environment
uv venv --python 3.10 hnet-env

# 3. Activate environment
source hnet-env/bin/activate

# 4. Install from requirements.txt (CPU-only)
uv pip install -r requirements.txt

# 5. For full GPU support (Linux + CUDA only):
cd repo/hnet
uv pip install -e .
```

### Linux + CUDA Setup

For full functionality with GPU dependencies:

**Requirements:**
- Linux OS (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+ installed
- nvcc compiler available in PATH

**Installation:**
```bash
# Verify CUDA
nvcc --version

# Create environment
uv venv --python 3.10 hnet-env
source hnet-env/bin/activate

# Install with GPU dependencies
cd repo/hnet
uv pip install -e .
```

---

## File Artifacts

### Generated Files

1. **Environment Directory:**
   ```
   /private/tmp/Paper2Agent/HNet_Agent/hnet-env/
   ```

2. **Requirements File:**
   ```
   /private/tmp/Paper2Agent/HNet_Agent/requirements.txt
   ```

3. **Pytest Configuration:**
   ```
   /private/tmp/Paper2Agent/HNet_Agent/pytest.ini
   /private/tmp/Paper2Agent/HNet_Agent/conftest.py
   ```

4. **This Report:**
   ```
   /private/tmp/Paper2Agent/HNet_Agent/reports/environment-manager_results.md
   ```

---

## Package Statistics

### Installation Summary

- **Total packages installed:** 153
- **PyPI installations:** 151 (core dependencies and testing tools)
- **Local installations:** 1 (hnet package in editable mode)
- **Git URL installations:** 0 (GPU dependencies skipped on macOS)

### Dependency Categories

**Core ML/Scientific:**
- PyTorch, NumPy, SciPy, einops, optree

**Testing & Development:**
- pytest, pytest-asyncio, ipykernel, papermill, nbclient

**Utilities:**
- omegaconf, regex, packaging, ninja, setuptools

**MCP Framework:**
- fastmcp, mcp, starlette, uvicorn

---

## Usage Recommendations

### For Code Analysis & Structure Inspection

The current environment is suitable for:
- ✓ Reading and analyzing source code
- ✓ Understanding model architecture
- ✓ Extracting documentation and comments
- ✓ Static code analysis
- ✓ Testing configuration validation

### For Full Model Training/Inference

Requires Linux + CUDA environment:
- GPU-accelerated training with Mamba SSM
- Flash Attention optimizations
- Causal Convolution operations
- Triton kernel optimizations

---

## Conclusion

Successfully provisioned a CPU-compatible Python 3.10.19 virtual environment for the HNet research codebase. While GPU dependencies cannot be installed on macOS ARM, the environment provides:

1. Complete source code access
2. Functional test infrastructure
3. Core PyTorch and scientific computing stack
4. Reproducible requirements documentation

For full functionality including model training and inference, deployment on a Linux system with NVIDIA CUDA support is required.

---

**Environment Status:** ✓ Ready for code inspection and analysis
**GPU Status:** ✗ Requires Linux + CUDA for full functionality
**Reproducibility:** ✓ Fully documented with requirements.txt

