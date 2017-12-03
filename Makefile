test:
	PYTHONPATH=. MPLBACKEND="agg" coverage run --source peartree -m py.test --verbose
