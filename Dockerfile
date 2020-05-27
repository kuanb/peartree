FROM kuanb/peartree

RUN mkdir /code && \
	pip install --upgrade pip && \
    pip install numpy==1.18.4 scipy==1.4.1

WORKDIR /code

COPY requirements_dev.txt /code/
RUN pip install -r requirements_dev.txt

COPY requirements.txt /code/
RUN pip install -r requirements.txt