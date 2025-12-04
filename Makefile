# Safe Flight - Makefile
# 
# Primary entry point for all project commands.
# Run 'make help' to see available targets.

.PHONY: help setup sim hardware clean test analyze viz status

# Python interpreter - prefer conda environment if available
CONDA_PYTHON := $(shell command -v conda >/dev/null 2>&1 && conda run -n safe-flight which python 2>/dev/null)
PYTHON := $(if $(CONDA_PYTHON),conda run --no-capture-output -n safe-flight python,python3)

# Project directories
SCRIPTS_DIR := scripts
CONFIG_DIR := config

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help:
	@echo ""
	@echo "$(BLUE)Safe Flight - Vision-Based Obstacle Avoidance$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make setup        - Set up environment and download models"
	@echo "  make setup-models - Download ML models only"
	@echo "  make verify       - Verify environment setup"
	@echo ""
	@echo "$(GREEN)Simulation:$(NC)"
	@echo "  make sim          - Launch SITL simulation (default world)"
	@echo "  make sim-open     - Launch with open world"
	@echo "  make sim-headless - Launch without GUI (faster)"
	@echo ""
	@echo "$(GREEN)Hardware:$(NC)"
	@echo "  make hardware     - Launch hardware flight (real drone)"
	@echo "  make preflight    - Run preflight checks only"
	@echo ""
	@echo "$(GREEN)Analysis:$(NC)"
	@echo "  make analyze      - Analyze latest flight log"
	@echo "  make viz          - Generate visualizations"
	@echo "  make status       - Show storage status"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test              - Run all tests"
	@echo "  make test-quick        - Run tests without hardware/slow markers"
	@echo "  make test-hardware     - Run hardware tests (requires drone)"
	@echo "  make test-hardware-quick - Hardware tests without flight"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@echo "  make clean        - Interactive cleanup"
	@echo "  make clean-all    - Clean logs, viz, and cache"
	@echo "  make clean-cache  - Clean Python cache only"
	@echo ""

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

setup:
	@echo "$(BLUE)Setting up Safe Flight environment...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/setup.py

setup-models:
	@echo "$(BLUE)Downloading ML models...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/setup.py --models

verify:
	@echo "$(BLUE)Verifying environment setup...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/setup.py --verify

#------------------------------------------------------------------------------
# Simulation
#------------------------------------------------------------------------------

sim:
	@echo "$(BLUE)Launching SITL simulation...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_sim.py

sim-open:
	@echo "$(BLUE)Launching SITL with open world...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_sim.py --world open --viz

sim-headless:
	@echo "$(BLUE)Launching SITL in headless mode...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_sim.py --no-gui

sim-viz:
	@echo "$(BLUE)Launching SITL with visualization...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_sim.py --viz

# Custom simulation with arguments
# Usage: make sim-custom ARGS="--world open --goal 5 0 1"
sim-custom:
	$(PYTHON) $(SCRIPTS_DIR)/launch_sim.py $(ARGS)

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

hardware:
	@echo "$(BLUE)Launching hardware flight...$(NC)"
	@echo "$(YELLOW)⚠️  Ensure safety precautions are in place!$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_hardware.py

hardware-hover:
	@echo "$(BLUE)Launching hover test (no vision)...$(NC)"
	@echo "$(YELLOW)⚠️  Ensure safety precautions are in place!$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_hardware.py --hover --duration 10

preflight:
	@echo "$(BLUE)Running preflight checks...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_hardware.py --preflight

# Custom hardware launch with arguments
# Usage: make hardware-custom ARGS="--uri radio://0/80/2M/E7E7E7E7E7"
hardware-custom:
	$(PYTHON) $(SCRIPTS_DIR)/launch_hardware.py $(ARGS)

#------------------------------------------------------------------------------
# Analysis
#------------------------------------------------------------------------------

analyze:
	@echo "$(BLUE)Analyzing latest flight log...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_sim.py --analyze

viz:
	@echo "$(BLUE)Generating visualizations...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/launch_sim.py --viz

status:
	@echo "$(BLUE)Storage status...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/cleanup.py --status

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

test:
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v

test-quick:
	@echo "$(BLUE)Running quick tests (no hardware, no slow)...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "not slow and not hardware"

test-hardware:
	@echo "$(BLUE)Running hardware tests...$(NC)"
	@echo "$(YELLOW)⚠️  Requires Crazyflie and Crazyradio connected!$(NC)"
	$(PYTHON) -m pytest tests/test_hardware_integration.py -v --hardware

test-hardware-quick:
	@echo "$(BLUE)Running quick hardware tests (no flight)...$(NC)"
	$(PYTHON) -m pytest tests/test_hardware_integration.py -v --hardware -m "not slow"

test-coverage:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

#------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------

clean:
	@echo "$(BLUE)Interactive cleanup...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/cleanup.py

clean-all:
	@echo "$(BLUE)Cleaning all logs, visualizations, and cache...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/cleanup.py --all

clean-logs:
	@echo "$(BLUE)Cleaning log files...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/cleanup.py --logs

clean-viz:
	@echo "$(BLUE)Cleaning visualization files...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/cleanup.py --viz

clean-cache:
	@echo "$(BLUE)Cleaning Python cache...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/cleanup.py --cache

clean-models:
	@echo "$(BLUE)Removing downloaded models...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/setup.py --clean

# Keep recent files
# Usage: make clean-old KEEP=5
clean-old:
	$(PYTHON) $(SCRIPTS_DIR)/cleanup.py --all --keep-latest $(or $(KEEP),5)

# Dry run cleanup
clean-dry:
	$(PYTHON) $(SCRIPTS_DIR)/cleanup.py --all --dry-run

#------------------------------------------------------------------------------
# Development
#------------------------------------------------------------------------------

# Format code with black and isort
format:
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Lint code with ruff
lint:
	@echo "$(BLUE)Linting code...$(NC)"
	ruff check src/ tests/ scripts/

# Type check with mypy
typecheck:
	@echo "$(BLUE)Type checking...$(NC)"
	mypy src/

#------------------------------------------------------------------------------
# Documentation
#------------------------------------------------------------------------------

docs:
	@echo "$(BLUE)Documentation files:$(NC)"
	@echo "  - README.md"
	@echo "  - docs/DOCUMENTATION.md"
	@echo "  - docs/DEPLOYMENT_PLAN.md"
	@echo "  - docs/DEVELOPMENT_LOG.md"
