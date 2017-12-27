FROM calthorpeanalytics/python3-geo:3.6.3-1.0.0

RUN mkdir -p /provisioning/peartree
COPY . /provisioning/peartree

# can now install Peartree via repo
RUN cd /provisioning/peartree && \
    pip install .