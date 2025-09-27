# Fase 1: Scegliamo un'immagine di base con Python
FROM python:3.9-slim

# Fase 2: Creiamo una cartella di lavoro all'interno del container
WORKDIR /app

# Fase 3: Copiamo il file delle dipendenze
COPY requirements.txt .

# Fase 4: Installiamo le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Fase 5: Copiamo tutti i file del nostro progetto nel container
# (proxy_server.py, index.html, e la cartella data/)
COPY . .

# Fase 6: Diciamo a Docker che il nostro server ascolta sulla porta 8001
EXPOSE 8001

# Fase 7: Il comando che verrà eseguito all'avvio del container
CMD ["python", "proxy_server.py"]