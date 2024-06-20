# DeepViscosity
DeepViscosity is an ensemble deep learning ANN model developed to predict high-concentration monoclonal antibody viscosity classes (Low <= 20 cP, High > 20 cP). The model utilized 30 spatial properties (descriptors) obtained from DeepSP surrogate model as features for training. It was trained based on 229 mAbs.

# How to use DeepViscosity to Predict Low or High Class Viscosity
- Prepare your input file according to the format DeepViscosity_input.csv
- Run the notebook file DeepViscosity_predictor.ipynb
- DeepViscosity Classes (as well as the DeepSP spatial properties) for sequences inputed, would be polulated and saved to a csv file.

# Citation
...
