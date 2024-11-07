# DeepViscosity
DeepViscosity is an ensemble deep learning ANN model developed to predict high-concentration monoclonal antibody viscosity classes (Low <= 20 cP, High > 20 cP). The model utilized 30 spatial properties (descriptors) obtained from DeepSP surrogate model as features for training. It was trained based on 229 mAbs.

# How to use DeepViscosity to Predict Low or High Class Viscosity
## Google colab notebook
- Prepare your input file according to the format DeepViscosity_input.csv
- Run the notebook file DeepViscosity_predictor.ipynb
- DeepViscosity Classes (as well as the DeepSP spatial properties) for sequences inputed, will be populated and saved to a csv file - DeepViscosity_classes.csv.

## Linux environment 
- create an environment and install necessary package
	conda create -n deepViscosity python=3.9.13
	source activate deepViscosity
	conda install -c bioconda anarci
	pip install keras==2.11.0 tensorflow-cpu==2.11.0 scikit-learn==1.0.2 pandas numpy==1.26.4 joblib dill

- Prepare your input file according to the format DeepViscosity_input.csv
- Run the python file deepviscosity_predictor.py
- DeepViscosity Classes (as well as the DeepSP spatial properties) for sequences inputed, will be obtained and saved to a csv file - DeepViscosity_classes.csv.

# Citation
...