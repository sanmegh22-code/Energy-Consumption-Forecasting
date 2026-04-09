from flask import Flask, render_template, request, send_file
import pandas as pd, os, pickle, generate_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
app.config["UPLOAD_FOLDER"] = "uploads"

# Global state
state = {'model': None, 'accuracy': None, 'train_size': None, 'test_size': None,
         'actual_values': None, 'predicted_values': None, 'last_prediction': None, 'dataset_hints': None, 'table_data': None}

if os.path.exists("model.pkl"):
    try:
        with open("model.pkl", "rb") as f:
            state.update(pickle.load(f))
    except: pass

def render_idx(**kw):
    return render_template("index.html", **{**state, 'predicted_value': None, 'actual_value': None, 
                                            'error': None, 'error_percent': None, 'result': None, 'message': None, **kw})

def train_model(data, model_type="LinearRegression"):
    state['dataset_hints'] = {col: f"{data[col].min():.1f} - {data[col].max():.1f}" for col in ["T1", "RH_1", "T2", "RH_2"]}
    X, y = data[["T1", "RH_1", "T2", "RH_2"]], data["Appliances"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    act_list = y_test[:20].tolist()
    pred_list = y_pred[:20].tolist()
    tbl = [{"actual": round(a, 2), "predicted": round(p, 2), "error": round(abs(a-p), 2)} for a, p in zip(act_list, pred_list)]
    
    state.update({
        'model': model, 'accuracy': round(r2_score(y_test, y_pred), 3),
        'train_size': len(X_train), 'test_size': len(X_test),
        'actual_values': act_list, 'predicted_values': pred_list,
        'table_data': tbl
    })
    generate_report.create_training_graph(state['actual_values'], state['predicted_values'])
    with open("model.pkl", "wb") as f: pickle.dump(state, f)

@app.route("/")
def home(): return render_idx()

@app.route("/train_predefined", methods=["POST"])
def train_predefined():
    m_type = request.form.get("model_type", "LinearRegression")
    merged_file = "energydata_master.csv"
    data_file = merged_file if os.path.exists(merged_file) else "energydata_complete.csv"
    train_model(pd.read_csv(data_file, encoding="latin1"), m_type)
    return render_idx(message=f"{m_type} trained successfully on current knowledge base!")

@app.route("/upload", methods=["POST"])
def upload():
    file, m_type = request.files.get("dataset"), request.form.get("model_type", "LinearRegression")
    if not file: return render_idx(message="No file selected")
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)
    try:
        new_data = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        base_file = "energydata_complete.csv"
        merged_file = "energydata_master.csv"
        current_data = pd.read_csv(merged_file if os.path.exists(merged_file) else base_file, encoding="latin1")
        
        # Combine datasets to "learn from it" continually
        combined_data = pd.concat([current_data, new_data], ignore_index=True)
        combined_data.to_csv(merged_file, index=False)
        
        train_model(combined_data, m_type)
        return render_idx(message=f"Dataset appended! Model upgraded and learned from expanded data using {m_type}.")
    except Exception as e: return render_idx(message=f"Error: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    if not state.get('model'): return render_idx(message="Train model first!")
    try: inputs = [float(request.form[k]) for k in ["T1", "RH_1", "T2", "RH_2"]]
    except: return render_idx(message="Invalid inputs.")
    
    energy_wh = round(state['model'].predict([inputs])[0], 2)
    state['last_prediction'] = energy_wh
    
    act_val = round(state['actual_values'][0], 2) if state['actual_values'] else 0
    err = round(abs(act_val - energy_wh), 2)
    err_pct = round((err / act_val) * 100, 2) if act_val else 0
    lvl = "Low" if energy_wh < 100 else "Moderate" if energy_wh < 300 else "High"
    
    generate_report.create_prediction_graph(energy_wh)
    with open("model.pkl", "wb") as f: pickle.dump(state, f)
    
    return render_idx(result=f"Usage Level: {lvl} Usage", predicted_value=energy_wh, 
                      actual_value=act_val, error=err, error_percent=err_pct)

@app.route("/download_report")
def download_report():
    elements = [Paragraph("Energy Consumption Report", getSampleStyleSheet()["Title"]), Spacer(1, 20)]
    elements.append(Table([["Metric", "Value"], ["R² Score", str(state['accuracy'])], ["Train Size", str(state['train_size'])],
                           ["Test Size", str(state['test_size'])], ["Last Prediction", str(state['last_prediction'])]]))
    for img, title in [("prediction_graph.png", "Prediction Graph"), ("training_graph.png", "Actual vs Predicted")]:
        if os.path.exists(img): elements.extend([Spacer(1, 20), Paragraph(title, getSampleStyleSheet()["Heading2"]), Image(img, 400, 250)])
    SimpleDocTemplate("report.pdf").build(elements)
    return send_file("report.pdf", as_attachment=True)

if __name__ == "__main__": app.run(debug=True, port=5000)