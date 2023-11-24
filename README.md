# Mispronunciation detection App

### About

This repo contains the code for the Sound and music computing final project.

This project is essentially a Mispronunciation detection system which can help L2 learners of English understand the pronunciation problems

It has three components to it:
1. <u> Frontend (React code) </u>
- Response & Intuitive user interface which gamifies the process of learning new words and sentences
- Has a "virtual assistant" (built using Azure TTS) which can show the lip movement animation for a given sentence thereby enabling users to easily understand the pronunciation

2. <u> Backend (Fast api) </u>
- Exposes api's needed for user login and progress state
- Connects to a backend DB which contains user and word-phoneme information needed for constructing sentences
- ML Inference API which accepts audio from UI as input and returns the predicted ARPABET phoneme sequences and provides a score


3. <u> ML Model (Wave2Vec2 model trained on L2 Arctic using Speechbrain dataset) </u>
- Model trained on L2 Arctic dataset for a few specific accents ( refer to ./src/ml/util/prepare_data.ipynb for more details) on Google Colab
- The checkpoints are present in <a href="https://drive.google.com/drive/folders/1-KvGwl8OBnUelgKWP5ex3BWNP54ruJu8?usp=sharing">this drive link</a>. Download the entire "results" folder and place it in the root level (/src/results)
- The checkpoints need to placed there in order for the server to boot up

### Frontend Setup

Refer to <a href = "https://github.com/TheDorkKnightRises/smc-pronunciation-app">this github repository</a> and refer to the readme file there for running the UI locally.

### ML Setup

1. Download the checkpoint from the drive link mentinoed earlier and place the "results" folder under "src" folder
2. For retraining each of the model again here are the steps needed:
* Download the L2 arctic dataset and upload to your personal google drive in a specific location inside the folder "dataset"
* Run the /src/ml/util/prepare_data.ipynb as a colab notebook in order to read the dataset and create train, test, val splits along with some phoneme preprocessing.
* Once this script runs successfully, a new "data" folder will be created with the final train, test & val splits needed for training the model
* Refer to the individual ipynb files which can be found under "./src/ml/train/X/X_train.ipynb" for training specific models in colab (out of Wav2Vec2, Hubert & Whisper we found Wav2Vec2 to be the best in terms of fast inference, performance and easy integration with the app)
* In each of these ipynb files ensure that the appropriate train yaml file from "./src/ml/config/X/train.yaml" is uploaded in the correct drive path as mentioned in the ipynb when creating the dataloaders.
* Once the models are trained the checkpoints will be populated in a folder called "results" which can then be downloaded locally and placed in your server (in the location as mentioned earlier) to ensure that the backend inference api can work seamlessly

### Backend Setup


1. Run the following commands for installing the required packages:

```
cd src

pip install -r .\requirements.txt
```

2. Use /src/db_schema.sql and create a postgres db in the cloud
3. Use the DB connection string and replace the string 'POSTGRES_DB_STRING_TO_BE_REPLACED' found inside /src/infra/db
4. cd into the src directory run this command:

```
python -m uvicorn main:app --reload
```


