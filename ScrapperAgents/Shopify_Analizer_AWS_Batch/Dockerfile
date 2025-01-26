FROM python:3.10-bullseye

WORKDIR /app
COPY ./analizador_shopify/requirements.txt /app
RUN pip install -r requirements.txt

COPY ./analizador_shopify/ /app

CMD ["python", "app.py"]


