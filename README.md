# Speech-Understanding-Major-Project (Speech-Enhancement-Using-Diffusion)

## Description
This project is build intended to submit as Major Project for Speech Understanding course at IIT, Jodhpur. 

## Models

| % Data Trained On | Model Name        | Download Link                                            |
|-------------------|-------------------|----------------------------------------------------------|
| <span rowspan="4">1%</span> | Model on 20% data         | [Download](https://drive.google.com/file/d/1-xkDA620XwlOrHlciGdc2QwTVNSFgoDQ/view?usp=sharing)   |


## Project Setup and Uses Steps
### Installation
To install and run this project, follow these steps:

1. Clone the repository: ```git clone https://github.com/parasharharsh16/Speech-Enhancement-using-Diffusion.git```
2. Navigate into the project directory: ```cd Speech-Enhancement-using-Diffusion```
3. Create conda (use miniconda or anaconda) environment ```conda create --prefix ./.venv python=3.9```
4. Activate conda environment ```conda activate ./.venv```
3. Install the required dependencies: ```pip install -r requirements.txt```
4. Download models from the links provided below and place them inside models folder. NB - don't change the name of model files.
5. Change `mode` to test in `params.py` to run just evaluations(test).

### Prepare Dataset
To download the "VCTK-Corpus" dataset, please follow the below link:
`https://www.kaggle.com/datasets/krupapatil/vctk-dataset`

- Unzip the downloaded .zip file and paste the VCTK-Corpus dataset to "data" folder.

### Run the code
1. Go to param.py and change `mode` to  `test` to run in evaluation mode, if intending to train then you may keep it to `train`
2. Run the main script: `python main.py`

### Results
- Training results can be viewed using [wandb.ai](https://wandb.ai/m22aie/Speech%20Enhancement%20using%20Diffusion?nw=nwuserpscss).
- `output/wav` folder contains generated cleaned audio files.
- `output/spectogram` folder contains spectograms from a random test set. 

## Team Members

- Prateek Singhal (m22aie215@iitj.ac.in)
- Prabha Sharma (m22aie224@iitj.ac.in)
- Harsh Parashar (m22aie210@iitj.ac.in)
