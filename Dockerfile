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

RUN echo "Installing GEOS library..." && \
    mkdir -p /provisioning/geos && \
    cd /provisioning/geos && \
    curl -# -O http://download.osgeo.org/geos/geos-3.5.1.tar.bz2 && \
    tar -xjf geos-3.5.1.tar.bz2 && \
    cd geos-3.5.1 && \
    ./configure && \
    make -j$(python -c 'import multiprocessing; print(multiprocessing.cpu_count())') && \
    make install && \
    ldconfig -v && \
    rm -rf /provisioning/geos*

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

RUN mkdir /code && \
    pip install numpy==1.12.1 --src /usr/local/src
WORKDIR /code

COPY requirements.txt /code/
RUN pip install -r requirements.txt --src /usr/local/src --exists-action=w

COPY requirements_dev.txt /code/
RUN pip install -r requirements_dev.txt --src /usr/local/src --exists-action=w