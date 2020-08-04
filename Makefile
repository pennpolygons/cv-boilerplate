TEST_PATH=./tests

init:
	pip install -e .

init-conda:
	yes | conda create --name cvb python=3.6 ;\
	conda activate cvb ;\
	pip install -e .

lint:
	black research

test:
	py.test --verbose --color=yes $(TEST_PATH)

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	name '*~' -exec rm --force  {}

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

.PHONY: init test clean-pyc clean-build