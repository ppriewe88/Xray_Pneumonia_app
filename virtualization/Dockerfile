# import python base image
FROM python:3.12-slim

# create working directory (root)
WORKDIR /app

# copy requirements into root dir
COPY requirements.txt .

# install requirements with pip
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install tensorflow-cpu==2.16.1

# copy all the project contents in /app
COPY . .

# open communication ports
EXPOSE 8000 8080

# command ran during the docker building process
CMD ["bash", "run_app_docker.sh"]