PROJECT_NAME = QDYN
PACKAGES =  pip numpy matplotlib scipy sympy ipython bokeh pytest
TESTPYPI = https://testpypi.python.org/pypi

TESTOPTIONS = --doctest-modules
TESTS = QDYN tests slow_tests
# You may redefine TESTS to run a specific test. E.g.
#     make test TESTS="tests/test_io.py"

develop:
	pip install -e .[dev]

install:
	pip install .

uninstall:
	pip uninstall $(PROJECT_NAME)

sdist:
	python setup.py sdist

upload:
	python setup.py register
	python setup.py sdist upload

test-upload:
	python setup.py register -r $(TESTPYPI)
	python setup.py sdist upload -r $(TESTPYPI)

test-install:
	pip install -i $(TESTPYPI) $(PROJECT_NAME)

clean:
	@rm -rf build
	@rm -rf __pycache__
	@rm -rf dist
	@rm -f *.pyc
	@rm -rf QDYN.egg-info
	@rm -f QDYN/*.pyc
	@rm -f QDYN/prop/*.pyc
	@rm -f tests/*.pyc
	@rm -rf tests/__pycache__
	@rm -f QDYN/__git__.py
	@rm -f test_octconvergences.html
	@rm -f tests/result_images/*

.venv/py27/bin/py.test:
	@conda create -y -m -p .venv/py27 python=2.7 $(PACKAGES)
	@.venv/py27/bin/pip install -e .[dev]

.venv/py33/bin/py.test:
	@conda create -y -m -p .venv/py33 python=3.3 $(PACKAGES)
	@.venv/py33/bin/pip install -e .[dev]

.venv/py34/bin/py.test:
	@conda create -y -m -p .venv/py34 python=3.4 $(PACKAGES)
	@.venv/py34/bin/pip install -e .[dev]

test27: .venv/py27/bin/py.test
	$< -v $(TESTOPTIONS) $(TESTS)

test33: .venv/py33/bin/py.test
	$< -v $(TESTOPTIONS) $(TESTS)

test34: .venv/py34/bin/py.test
	$< -v $(TESTOPTIONS) $(TESTS)

test: test27 test33 test34

.PHONY: install develop uninstall upload test-upload test-install sdist clean \
test test27 test33 test34
