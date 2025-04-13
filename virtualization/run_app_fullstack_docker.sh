################################# NOTICE ##########################
# This script starts only the FRONTEND in the docker FRONTEND container!
###################################################################
# in backend: navigate to directory mlruns are stored
cd /app/backend/src/models/
# in backend: start mlflow server in docker (& tells it to run in the background)
mlflow server --host 0.0.0.0 --port 8080 &
# in backend: navigate to api directory
cd /app/backend/src/api
# in backend: start uvicorn server (api) in docker (& tells it to run in the background)
uvicorn api_server:app --host 0.0.0.0 --port 8000 &
# navigate to frontend directory 
cd /app/frontend
# in frontend: start react app
npm run start -- --host 0.0.0.0 --port 3000