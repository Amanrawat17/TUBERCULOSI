from flask import Flask, render_template, request
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
import pickle
import numpy as np
import io
import base64
import bz2  # Import bz2 for decompression

app = Flask(__name__)

def load_model(compressed_file_path):
    # Decompress and load the model
    with bz2.open(compressed_file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load pre-trained model
model_path = 'best_stacking_classifier (1).pkl.bz2'
model = load_model(model_path)

def drawing_molecule(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        img = Draw.MolToImage(mol)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64
    return None

def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    return np.zeros(n_bits)

def compute_all_descriptors(smiles):
    descriptor_names = [desc[0] for desc in Descriptors.descList[:208]]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(descriptor_names))
    descriptors = [Descriptors.__dict__[desc](mol) for desc in descriptor_names]
    return np.array(descriptors)

def model_predict(model, smiles):
    morgan = generate_morgan_fingerprint(smiles)
    descriptors = compute_all_descriptors(smiles)
    feature_vector = np.concatenate((morgan, descriptors)).reshape(1, -1)
    prediction = model.predict(feature_vector)
    confidence = model.predict_proba(feature_vector).max()
    return 'Active' if prediction[0] == 1 else 'Inactive', confidence

def molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    properties = {"Molecular Weight": None, "Chemical Formula": None}
    if mol is not None:
        properties["Molecular Weight"] = Descriptors.MolWt(mol)
        properties["Chemical Formula"] = rdMolDescriptors.CalcMolFormula(mol)
    return properties

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        smile_input = request.form.get("smile_input")
        if smile_input:
            try:
                mol_image = drawing_molecule(smile_input)
                properties = molecular_properties(smile_input)
                mol_weight = properties.get("Molecular Weight", "N/A")
                mol_formula = properties.get("Chemical Formula", "N/A")
                rf_pred, rf_conf = model_predict(model, smile_input)
                return render_template("index.html", 
                                       mol_image=mol_image,
                                       mol_weight=mol_weight,
                                       mol_formula=mol_formula,
                                       rf_pred=rf_pred,
                                       rf_conf=rf_conf * 100)
            except Exception as e:
                return render_template("index.html", error=f"An error occurred: {str(e)}")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
