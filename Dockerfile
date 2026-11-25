
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY app/requirements.txt .

# PyTorch is already installed in the base image.
RUN sed -i '/^torch$/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
