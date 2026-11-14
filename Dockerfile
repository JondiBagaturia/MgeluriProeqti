# Python 3.11 pinned: astropy/astroquery/reproject/numpy have no wheels yet
# for newer Pythons as of Nov 2026 (see PLAN.md).
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY config/ config/
COPY backend/ backend/
COPY frontend/ frontend/

WORKDIR /app/backend
EXPOSE 8000

# Stateless: no volumes needed. Horizontally scalable behind any load
# balancer — every request is independent, nothing is written to disk.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
