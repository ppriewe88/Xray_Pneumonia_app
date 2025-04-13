################################# NOTICE ##########################
# This script starts only the frontend in the docker frontend container!
################################# NOTICE ##########################
cd /app/frontend
# in frontend: start react app
npm run start -- --host 0.0.0.0 --port 3000