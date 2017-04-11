PROJECT_NAME = QDYN
PACKAGES =  pip numpy matplotlib scipy sympy ipython bokeh pytest coverage click sphinx
TESTPYPI = https://testpypi.python.org/pypi

TESTOPTIONS = --doctest-modules --cov=QDYN --cov-config .coveragerc -n auto
TESTS = QDYN tests
# You may redefine TESTS to run a specific test. E.g.
#     make test TESTS="tests/test_io.py"

help:
	@echo 'Makefile for qdynpylib                                                 '
	@echo '                                                                       '
	@echo 'Usage:                                                                 '
	@echo '   make develop       Install "editable" version of package            '
	@echo '   make install       Install package into current environment         '
	@echo '   make uninstall     Remove package from current environment          '
	@echo '   make upload        Upload package to pypi                           '
	@echo '   make test-upload   Upload package to testpypi                       '
	@echo '   make test-install  Install fromm testpypi                           '
	@echo '   make clean         Remove build files                               '
	@echo '   make test27        Test on Python 2.7                            	  '
	@echo '   make test33        Test on Python 3.3                               '
	@echo '   make test34        Test on Python 3.4                               '
	@echo '   make test          Test on all versions of Python                   '
	@echo '   make coverage      Generate coverage report htmlcov                 '

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
	@rm -rf QDYN/__pycache__
	@rm -f tests/*.pyc
	@rm -rf tests/__pycache__
	@rm -f QDYN/__git__.py
	@rm -f test_octconvergences.html
	@rm -f tests/result_images/*
	@rm -f .coverage

clean-doc:
	@rm -rf docs/build
	@rm -rf doc

distclean: clean clean-doc
	@rm -rf .venv

.venv/py27/bin/py.test:
	@conda create -y -m -p .venv/py27 python=2.7 $(PACKAGES)
	@.venv/py27/bin/pip install -e .[dev]

.venv/py33/bin/py.test:
	@conda create -y -m -p .venv/py33 python=3.3 $(PACKAGES)
	@.venv/py33/bin/pip install -e .[dev]

.venv/py34/bin/py.test:
	@conda create -y -m -p .venv/py34 python=3.4 $(PACKAGES)
	@.venv/py34/bin/pip install -e .[dev]

.venv/py35/bin/py.test:
	@conda create -y -m -p .venv/py35 python=3.5 $(PACKAGES)
	@.venv/py35/bin/pip install -e .[dev]

.venv/py36/bin/py.test:
	@conda create -y -m -p .venv/py36 python=3.6 $(PACKAGES)
	@.venv/py36/bin/pip install -e .[dev]

test27: .venv/py27/bin/py.test
	PYTHONHASHSEED=0 $< -v $(TESTOPTIONS) $(TESTS)

test33: .venv/py33/bin/py.test
	PYTHONHASHSEED=0 $< -v $(TESTOPTIONS) $(TESTS)

test34: .venv/py34/bin/py.test
	PYTHONHASHSEED=0 $< -v $(TESTOPTIONS) $(TESTS)

test35: .venv/py35/bin/py.test
	PYTHONHASHSEED=0 $< -v $(TESTOPTIONS) $(TESTS)

test36: .venv/py36/bin/py.test
	PYTHONHASHSEED=0 $< -v $(TESTOPTIONS) $(TESTS)

test: test27 test33 test34 test35 test36

doc: .venv/py35/bin/py.test docs/source/*.rst QDYN/*.py
	@rm -f docs/source/API/*
	$(MAKE) -C docs SPHINXBUILD=../.venv/py35/bin/sphinx-build html
	@ln -sf docs/build/html doc

coverage: test34
	@rm -rf htmlcov/index.html
	.venv/py34/bin/coverage html

.PHONY: install develop uninstall upload test-upload test-install sdist clean \
test test27 test33 test34 coverage
