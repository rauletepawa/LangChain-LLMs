FROM python:3.10-bullseye
WORKDIR /app
COPY requirements.txt .
COPY logo/ logo/
COPY pdf/ pdf/
COPY templates/ templates/
COPY static/ static/
RUN pip install -r requirements.txt
COPY application.py .

# Run your Flask application
EXPOSE 5000
CMD [ "flask", "run","--host","0.0.0.0","--port","5000"]


