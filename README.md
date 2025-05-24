# X-Ray_Pneumonia

This repository holds codes for a hobby / portfolio project.  
Currently worked on by Patrick and Andrei.
Train & test data taken from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.

---

### 🧠 Fullstack Project Highlights

- **Fullstack project integration of various topics/components**: Tensorflow models, backend business logic, FastAPI, MLFlow training and performance tracking, React frontend, shadow deployment

--- 



<p align="left">
  <strong><a href="https://youtu.be/aaeOJk1loig">📺 Watch short demo video on YouTube</a></strong>
</p>



<p align="left">
  <strong>🚀 Quick Project documentation</strong>  
</p>

Main functionalities:
  - Training (backend):
    - model training: own CNN model, transfer learning (MobileNet), fine tuning (MobileNet)
    - tracking of training perfomance with `mlflow`
  - Model administration & tracking (backend):
    - model registration with `mlflow`
    - csv logging of model performance on new, unseen data
    - supervision of and automated switch between two competing models ("champion" vs "challenger")
  - Frontend:
    - image upload and inferrence (pneumonia indication)
    - visualization of ML models' performance
    - embedded API documentation and model administration panel



<p align="left">
  <strong> 🤖 Workflow documentation</strong>  
</p>

![Achitecture   Workflows](https://github.com/user-attachments/assets/39e11f47-337b-4149-84dc-6315e4279f73)



<p align="left">
  <strong>🚀 How to start the app locally </strong>  
</p>

- Locally prepare Git:
  - Initialize empty git repository in a local folder (git init)
  - Specify this Github repository here as the remote origin
  - Pull content of this repo to your local one (git pull origin main)
- Create venv and install the required python packages (pip install -r requirements.txt).
- Build docker image from local repo (project directory) by running the following command from the project directory:
  - docker build -f virtualization/Dockerfile.fullstack -t xray_pneumonia .
- Then build and run docker container by running the following command from the project directory:
  - docker run -d --name xray_container -p 8000:8000 -p 8080:8080 -p 3000:3000 xray_pneumonia
- To access the GUI, open your browser and navigate to http://localhost:3000.
- Alternatively, one can also run the app without the frontend by running the run_app.sh bash script from the project directory and use your browser to navigate to:
  - http://127.0.0.1:8000/docs to access the FastAPI GUI
  - http://127.0.0.1:8080 to access the mlflow GUI


