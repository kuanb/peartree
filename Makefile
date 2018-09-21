test:
	PYTHONPATH=. MPLBACKEND="agg" coverage run --source peartree -m py.test --verbose

performance:
	PYTHONPATH=. MPLBACKEND="agg" pytest profiler/test_graph_assembly.py -s

notebook:
	docker-compose build
	mkdir -p ./notebooks
	docker-compose up notebook

docker-clean:
	docker network prune --force
	docker volume prune --force
	docker image prune --force

cprofile:
	pip install snakeviz
	python -m cProfile -o performance/cprof-output.py performance/run_etl.py
	snakeviz performance/cprof-output.py

install-graph-viz:
	apt-get update && \
	apt-get install xdg-utils && \
	apt-get install graphviz && \
	pip install pygraphviz

# Used this instead? apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7A80C8ED4FCCBE09
# pip install pycairo
install-graph-tool:
	echo "deb http://downloads.skewed.de/apt/stretch stretch main" | tee -a /etc/apt/sources.list && \
	echo "deb-src http://downloads.skewed.de/apt/stretch stretch main" | tee -a /etc/apt/sources.list && \
	apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25 && \
	apt-get update && \
	apt-get install python3-graph-tool

pypi-release:
	if [ -d dist ]; then rm -Rf dist; fi
	python setup.py bdist_wheel
	twine upload dist/*
