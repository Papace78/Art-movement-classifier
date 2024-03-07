# create an image from an environment
FROM python:3.10.6-buster

# COPY -> select the folder you need
COPY app_folder app_folder
COPY requirements.txt requirements.txt

# RUN run terminal command
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# install your package
COPY transfer_learning transfer_learning
COPY setup.py setup.py
RUN pip install .

EXPOSE 8000

# controls functinnality of the container
# uvicorn to control the server port
# local

CMD uvicorn app_folder.app_file:app --host 0.0.0.0

# Deploy to GCP
#CMD uvicorn app_folder.app_file:app --host 0.0.0.0 --port $PORT
