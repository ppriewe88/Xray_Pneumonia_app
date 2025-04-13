# navigate to directory mlruns are stored
cd backend/src/models/
# start mlflow server (& tells it to run in the background) 
mlflow server --host 127.0.0.1 --port 8080 &
# navigate to api directory
cd ../api/
# run api script
python api_server.py
# run react app locally
# NODE_OPTIONS=--openssl-legacy-provider npm start -- --host 0.0.0.0