FROM calthorpeanalytics/python3-geo:3.6.3-1.0.0

RUN mkdir /code
WORKDIR /code

RUN pip install numpy==1.12.1 --src /usr/local/src

COPY requirements.txt /code/
RUN pip install -r requirements.txt --src /usr/local/src --exists-action=w

COPY requirements_dev.txt /code/
RUN pip install -r requirements_dev.txt --src /usr/local/src --exists-action=w