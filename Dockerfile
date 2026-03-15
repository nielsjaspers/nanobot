FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p mike && touch mike/__init__.py && \
    uv pip install --system --no-cache . && \
    rm -rf mike

COPY mike/ mike/
RUN uv pip install --system --no-cache .

RUN mkdir -p /root/.mike

ENTRYPOINT ["nanobot"]
CMD ["gateway"]
