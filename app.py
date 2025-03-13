import gradio as gr
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load or train models
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("logistic_model.pkl", "rb") as f:
        logistic_model = pickle.load(f)

    with open("perceptron_model.pkl", "rb") as f:
        perceptron_model = pickle.load(f)

except FileNotFoundError:
    print("Training models...")
    df = pd.read_excel("Student-Employability-Dataset.xlsx", sheet_name="Data")
    X = df.iloc[:, 1:-2].values
    y = (df["CLASS"] == "Employable").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train_scaled, y_train)

    perceptron_model = Perceptron(random_state=42)
    perceptron_model.fit(X_train_scaled, y_train)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("logistic_model.pkl", "wb") as f:
        pickle.dump(logistic_model, f)
    with open("perceptron_model.pkl", "wb") as f:
        pickle.dump(perceptron_model, f)

# Prediction function
def predict_employability(name, ga, mos, pc, ma, sc, api, cs, model_choice):
    if not name:
        name = "Candidate"

    input_data = np.array([[ga, mos, pc, ma, sc, api, cs]])
    input_scaled = scaler.transform(input_data)

    model = logistic_model if model_choice == "Logistic Regression" else perceptron_model
    prediction = model.predict(input_scaled)

    return f"{name} is {'Employable 😊' if prediction[0] == 1 else 'Less Employable - Keep Improving! 💪'}"

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# AI-Powered Employability Assessment 🚀")
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name")
            ga = gr.Slider(1, 5, step=1, label="General Appearance")
            mos = gr.Slider(1, 5, step=1, label="Manner of Speaking")
            pc = gr.Slider(1, 5, step=1, label="Physical Condition")
            ma = gr.Slider(1, 5, step=1, label="Mental Alertness")
            sc = gr.Slider(1, 5, step=1, label="Self Confidence")
            api = gr.Slider(1, 5, step=1, label="Ability to Present Ideas")
            cs = gr.Slider(1, 5, step=1, label="Communication Skills")
            model_choice = gr.Radio(["Logistic Regression", "Perceptron"], label="Choose Model")
            evaluate_btn = gr.Button("Evaluate")

        with gr.Column():
            result_output = gr.Textbox(label="Employability Prediction")

    evaluate_btn.click(
        fn=predict_employability,
        inputs=[name, ga, mos, pc, ma, sc, api, cs, model_choice],
        outputs=result_output
    )

app.launch(share=True)
