FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY querypilot ./querypilot
COPY web ./web
COPY results ./results
RUN python -m querypilot.demo && chmod -R a-w data
# Run as an unprivileged user: the database files are read-only at the OS level too.
RUN useradd --create-home pilot
USER pilot
EXPOSE 8000
CMD ["uvicorn", "querypilot.server:app", "--host", "0.0.0.0", "--port", "8000"]
