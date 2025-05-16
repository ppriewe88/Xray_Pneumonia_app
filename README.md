# X-Ray_Pneumonia

This repository holds codes for a hobby / portfolio project.  
Currently worked on by Patrick and Andrei.

---

### 🧠 Fullstack Project Highlights

- **Fullstack project integration**: Tensorflow models, backend business logic, FastAPI, and React frontend

--- 

<p align="center">
  <strong>🚀 Project and workflow documentation: <em>SEE BELOW</em>!</strong>  
</p>
<p align="center">
  <strong><a href="https://youtu.be/aaeOJk1loig">📺 Watch short demo video on YouTube</a></strong>
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

![Achitecture   Workflows](https://github.com/user-attachments/assets/39e11f47-337b-4149-84dc-6315e4279f73)

