################################# NOTICE ##########################
# This script starts only the BACKEND in the docker BACKEND container!
################################# NOTICE ##########################
# in backend: navigate to directory where mlruns are stored
cd /app/backend/src/models/
# in backend: start mlflow server in docker (& tells it to run in the background)
mlflow server --host 0.0.0.0 --port 8080 &
# in backend: navigate to api directory
cd /app/backend/src/api
# in backend: start uvicorn server (api) in docker (& tells it to run in the background)
uvicorn api_server:app --host 0.0.0.0 --port 8000