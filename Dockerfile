FROM kuanb/peartree

RUN mkdir /code && \
    pip install numpy==1.12.1 --src /usr/local/src

WORKDIR /code

COPY requirements_dev.txt /code/
RUN pip install -r requirements_dev.txt

COPY requirements.txt /code/
RUN pip install -r requirements.txt