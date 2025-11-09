FROM python:3.13-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flask numpy scikit-learn joblib pandas

EXPOSE 5000

CMD ["python", "app.py"]
