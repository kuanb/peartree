========
Peartree
========

Peartree is a library for converting `GTFS <https://developers.google.com/transit/gtfs/>`__ feed schedules into a representative directed network graph. The tool uses `Partridge <https://github.com/remix/partridge>`__ to convert the target operator schedule data into `Pandas <https://github.com/pandas-dev/pandas>`__ dataframes and then `NetworkX <https://networkx.github.io/>`__ to hold the manipulated schedule data as a directed multigraph.