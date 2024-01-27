from flask import Flask, render_template, request, redirect, url_for
import pickle
from custom_metrics_module import r2_keras, attention
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

custom_objects = {'r2_keras': r2_keras, 'attention': attention}

app = Flask(__name__)
at = pickle.load(open("model/modelt.pkl", "rb"))

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint = np.array(fingerprint)
        return fingerprint  # Convert to numpy array
    else:
        return None

def reshapex(fp):
    n_features = 1
    x = fp.reshape(1, fp.shape[0], n_features)
    return x

def create_array(var1, var2, var3):
    my_array = np.array([var1, var2, var3], dtype=np.float32)  # Convert to numpy array with float32 dtype
    return my_array

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict_index():
    return render_template("predict.html")

@app.route("/predict/result", methods=["POST"])
def predict():
    smiles = request.form.get('smiles')
    homo = float(request.form.get('homo'))  # Convert to float
    lumo = float(request.form.get('lumo'))  # Convert to float
    energy = float(request.form.get('energy'))  # Convert to float

    ec = smiles_to_fingerprint(smiles)
    ecfp = reshapex(ec)
    qt = reshapex(create_array(homo, lumo, energy))

    if ecfp is not None:
        # Assuming your model expects a list of two inputs
        prediction = at.predict([ecfp, qt])
        prediction = prediction[0][0]
        print("SMILES:", smiles)
        print("Prediction:", prediction)
        # Pass the smiles and prediction to the template
        return render_template("predict.html", smiles=smiles, prediction=prediction)
    else:
        # Pass the smiles for displaying invalid input
        return render_template("predict.html", smiles=smiles, prediction="Invalid SMILES")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


