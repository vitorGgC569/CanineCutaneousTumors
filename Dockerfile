# --------- ESTÁGIO DE CONSTRUÇÃO (BUILDER) ---------
FROM python:3.14-slim as builder

# Variáveis de ambiente para o instalador e Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Instalação apenas das ferramentas necessárias para compilar pacotes complexos em C/C++
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openslide-tools \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cria e ativa um ambiente virtual para isolar as dependências
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copia os arquivos de dependência e instala no VENV (Aproveitando o cache de camadas)
COPY CATCH/requirements.txt /app/CATCH/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/CATCH/requirements.txt

COPY modern/requirements.txt /app/modern/requirements.txt
RUN pip install --no-cache-dir -r /app/modern/requirements.txt

# --------- ESTÁGIO FINAL (RUNTIME) ---------
FROM python:3.14-slim

# Variáveis de ambiente para execução
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Instala apenas as dependências de sistema estritamente necessárias para a execução
# Deixa de fora o "build-essential" (compiladores gcc, make, etc), reduzindo drasticamente o tamanho
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

# Puxa o ambiente virtual perfeitamente compilado do estágio anterior
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copia o código real da pesquisa
COPY . /app/

CMD ["bash"]
