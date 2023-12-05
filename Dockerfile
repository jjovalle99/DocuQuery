FROM python:3.11.6-slim
RUN useradd -m docuquery
USER docuquery
ENV HOME=/home/docuquery \
    PATH=/home/docuquery/.local/bin:$PATH \
    POETRY_VIRTUALENVS_IN_PROJECT=true
WORKDIR /app
COPY --chown=docuquery ./ ./
RUN pip install poetry --no-cache-dir && \
poetry install --only main --no-root --no-cache --no-interaction \
--no-ansi --no-cache
EXPOSE 7860
CMD [ "poetry", "run", "chainlit", "run", "app.py", "--port", "7860"]