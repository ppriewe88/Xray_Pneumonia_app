################################ backend stage ########################
# import python base image
FROM python:3.12-slim AS backend
# create backend in working directory
WORKDIR /app/backend
# copy requirements into backend root
COPY backend/requirements.txt .
# install requirements with pip
RUN pip install --no-cache-dir -r requirements.txt
# copy backend contents into backend root ("." =  /app/backend)
COPY backend/ .

################################ frontend stage ########################
# import node base image
FROM node:20-slim AS frontend
# create frontend in working directory
WORKDIR /app/frontend
# install required node modules (react, bootstrap, etc.) NOTE: using altered package.json, docker specific (no "set" command in npm run)!
COPY frontend/package.docker.json ./package.json
COPY frontend/package-lock.json ./package-lock.json
# install modules
RUN npm install --production
# copy frontend contents into frontend root ("." =  /app/frontend)
COPY frontend/ .

################################ final stage ##########################
# use python as base image
FROM python:3.12-slim
# install node again
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
# copy backend and frontend
COPY --from=backend /app/backend /app/backend
COPY --from=frontend /app/frontend /app/frontend

################################ ports and run configuration ##########
# open communication ports
EXPOSE 8000 8080 3000

# command ran during the docker building process
CMD ["bash", "run_app_docker.sh"]