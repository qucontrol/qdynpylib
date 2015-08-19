install:
	pip install -Ie .

uninstall:
	pip uninstall QDYN

sdist:
	python setup.py sdist

clean:
	@rm -rf build
	@rm -rf dist
	@rm -f *.pyc
	@rm -rf QDYN.egg-info
	@rm -f QDYN/*.pyc
	@rm -f QDYN/prop/*.pyc
	@rm -f tests/*.pyc
	@rm -f QDYN/__git__.py
	@rm -f test_octconvergences.html
	@rm -f tests/result_images/*

PACKAGES =  pip nose numpy matplotlib scipy sympy ipython bokeh

.venv/py27/bin/python:
	@conda create -y -m -p .venv/py27 python=2.7 $(PACKAGES)
	@.venv/py27/bin/pip install -e .

.venv/py33/bin/python:
	@conda create -y -m -p .venv/py33 python=3.3 $(PACKAGES)
	@.venv/py33/bin/pip install -e .

.venv/py34/bin/python:
	@conda create -y -m -p .venv/py34 python=3.4 $(PACKAGES)
	@.venv/py34/bin/pip install -e .

test27: .venv/py27/bin/python
	.venv/py27/bin/python run_tests.py

test33: .venv/py33/bin/python
	.venv/py33/bin/python run_tests.py

test34: .venv/py34/bin/python
	.venv/py34/bin/python run_tests.py


test: test27 test33 test34

.PHONY: install uninstall sdist clean test test27 test33 test34
