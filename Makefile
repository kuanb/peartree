test:
	PYTHONPATH=. MPLBACKEND="agg" coverage run --source peartree -m py.test --verbose

performance:
	PYTHONPATH=. MPLBACKEND="agg" pytest profiler/test_graph_assembly.py -s

notebook:
	docker pull tiagopeixoto/graph-tool
	# Will need to then run this: jupyter notebook --ip 0.0.0.0
	docker run -p 8888:8888 -p 6006:6006 -it -u user -w /home/user tiagopeixoto/graph-tool bash

