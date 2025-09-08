FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
COPY ./ ./

RUN pip3 install -r requirements.txt

ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]