FROM python:3.6-stretch

RUN mkdir -p /provisioning
WORKDIR /provisioning

# Install OS dependencies
RUN apt-get update && apt-get install -y \
      build-essential \
      dialog \
      curl \
      less \
      nano \
      unzip \
      vim \
      gcc \
      libgeos-dev \
      zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Get libgeos for Python Shapely package
# https://trac.osgeo.org/geos/
ARG GEOS_VERSION=3.6.2
RUN echo "Installing GEOS libraries..." && \
    mkdir -p /provisioning/geos && \
    cd /provisioning/geos && \
    curl -# -O http://download.osgeo.org/geos/geos-${GEOS_VERSION}.tar.bz2 && \
    tar -xjf geos-${GEOS_VERSION}.tar.bz2 && \
    cd geos-${GEOS_VERSION} && \
    ./configure && \
    make -j$(python -c 'import multiprocessing; print(multiprocessing.cpu_count())') && \
    make install && \
    ldconfig -v && \
    rm -rf /provisioning/geos*

# Get ESRI FileGDB libraries for Fiona/Geopandas Python packages
# http://appsforms.esri.com/products/download/
ARG FILEGDB_VERSION=1_4
RUN echo "Installing ESRI FileGDB libraries..." && \
    mkdir -p /provisioning/filegdb && \
    curl -# -o filegdb_api_${FILEGDB_VERSION}-64.tar.gz https://www.dropbox.com/s/dti2x6ydibyfs68/filegdb_api_1_2-64.tar.gz?dl=1 && \
    tar -zxvf filegdb_api_${FILEGDB_VERSION}-64.tar.gz && \
    cp -r FileGDB_API-64/lib/* /usr/local/lib && \
    cp -r FileGDB_API-64/include/* /usr/local/include && \
    ldconfig -v && \
    rm -rf /provisioning/filegdb* /provisioning/FileGDB*

# Compile GDAL with FileGDB support for Fiona/Geopandas Python packages
RUN echo "Installing GDAL libraries..." && \
    mkdir -p /provisioning/gdal && \
    cd /provisioning/gdal && \
    curl -# -o gdal-2.2.1.tar.gz http://download.osgeo.org/gdal/2.2.1/gdal-2.2.1.tar.gz && \
    tar -zxvf gdal-2.2.1.tar.gz && \
    cd /provisioning/gdal/gdal-2.2.1 && \
    ./configure --prefix=/usr/ --with-python --with-fgdb && \
    make -j$(python -c 'import multiprocessing; print(multiprocessing.cpu_count())') && \
    make install && \
    rm -rf /provisioning/gdal*

RUN echo "Installing Spatial Index library..." && \
    mkdir -p /provisioning/spatialindex && \
    cd /provisioning/spatialindex && \
    curl -# -O http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz && \
    tar -xzf spatialindex-src-1.8.5.tar.gz && \
    cd spatialindex-src-1.8.5 && \
    ./configure --prefix=/usr/local && \
    make -j$(python -c 'import multiprocessing; print(multiprocessing.cpu_count())') && \
    make install && \
    ldconfig && \
    rm -rf /provisioning/spatialindex*

RUN echo "Installing Proj4 library..." && \
    mkdir -p /provisioning/proj4 && \
    cd /provisioning/proj4 && \
    curl -# -O http://download.osgeo.org/proj/proj-4.9.3.tar.gz && \
    tar -xzf proj-4.9.3.tar.gz && \
    cd proj-4.9.3 && \
    ./configure && \
    make -j$(python -c 'import multiprocessing; print(multiprocessing.cpu_count())') && \
    make install && \
    ldconfig -v && \
    rm -rf /provisioning/proj4

# basemap (incorrectly) requires numpy to be installed *before* installing it
RUN pip install --upgrade numpy && \
    echo "Installing Basemap plotting library..." && \
    mkdir -p /provisioning/matplotlib-basemap && \
    cd /provisioning/matplotlib-basemap && \
    curl -# -o basemap-1.0.7rel.tar.gz https://codeload.github.com/matplotlib/basemap/tar.gz/v1.0.7rel && \
    tar -xzf basemap-1.0.7rel.tar.gz && \
    cd basemap-1.0.7rel && \
    python setup.py install && \
    rm -rf /provisioning/matplotlib-basemap