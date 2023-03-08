FROM python:3.8-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY pickle_files/ /deploy/pickle_files/
COPY prediction_models/ /deploy/prediction_models/
COPY ./data.csv /deploy/
COPY ./utils.py /deploy/

WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]