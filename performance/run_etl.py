import os
import tempfile
from urllib.request import urlopen

import partridge as ptg
import peartree as pt

pt.utilities.config(log_console=True)

if __name__ == '__main__':
    '''
    Run from command line via:
    python -m cProfile -o profile/cprof-output.py performance/run_etl.py

    And visualize via:
    snakeviz profile/cprof-output.py
    '''
    url_path = 'http://www.actransit.org/wp-content/uploads/GTFS_Fall17.zip'
    tempdir = tempfile.mkdtemp()
    filepath = os.path.join(tempdir, 'perf_gtfs.zip')

    response = urlopen(url_path)
    zipcontent = response.read()
    with open(filepath, 'wb') as f:
        f.write(zipcontent)

    feed = ptg.get_representative_feed(filepath)

    start = 7 * 60 * 60  # 7:00 AM
    end = 10 * 60 * 60  # 10:00 AM
    G = pt.load_feed_as_graph(feed, start, end, interpolate_times=True)
