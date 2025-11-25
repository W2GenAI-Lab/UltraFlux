FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ULTRAFLUX_DEVICE=cuda \
    TRANSFORMERS_CACHE=/app/.cache \
    HF_HOME=/app/.cache

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]

