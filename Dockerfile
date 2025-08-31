FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-geometric
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p uploads static/temp templates

EXPOSE 5000
CMD ["python", "app.py"]