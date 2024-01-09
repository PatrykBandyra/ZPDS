FROM python:3.11

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt  # Necessary for cv2 to work

COPY ./app /code/app

ENV PYTHONPATH /code/app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]