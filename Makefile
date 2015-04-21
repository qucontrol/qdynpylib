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

test:
	python run_tests.py

.PHONY: install uninstall sdist clean
