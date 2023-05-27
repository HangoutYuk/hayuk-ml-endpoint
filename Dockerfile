FROM python:3.11.3-slim-buster

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT 8000
EXPOSE 8000

CMD ["python", "app.py"]
