.PHONY: install test lint format clean build examples

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-bench:
	pytest tests/benchmarks/ -v -s

lint:
	ruff check nesy/ tests/
	mypy nesy/ --ignore-missing-imports

format:
	ruff format nesy/ tests/ examples/

examples:
	python examples/basic_reasoning.py
	python examples/medical_diagnosis.py
	python examples/continual_learning.py
	python examples/edge_deployment.py
	python examples/shadow_demo.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info

build:
	python -m build

serve:
	python -m nesy.deployment.server.app
