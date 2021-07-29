FROM python:3.9.6
WORKDIR /app
COPY . .
RUN pip install -r requirements-prod.txt
RUN useradd appuser && chown -R appuser /app
USER appuser
ENTRYPOINT ["uwsgi", "app.ini"]
EXPOSE 8080