FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8503
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]