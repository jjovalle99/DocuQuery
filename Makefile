format:
	isort *.py
	black *.py
	isort **/*.py
	black **/*.py
	isort src/openai_utils/*.py
	black src/openai_utils/*.py

pycache:
	find ./ -type d -name '__pycache__' -exec rm -rf {} +
