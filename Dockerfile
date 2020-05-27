FROM kuanb/peartree

RUN mkdir /code && \
	pip install --upgrade pip && \
    pip install numpy==1.14.0 scipy==1.0.0

WORKDIR /code

COPY requirements_dev.txt /code/
RUN pip install -r requirements_dev.txt

COPY requirements.txt /code/
RUN pip install -r requirements.txt