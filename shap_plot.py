import shap
import joblib
import matplotlib.pyplot as plt
import os

def generate_shap_plot(input_data):
    model = joblib.load('model/model.pkl')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Generate summary plot
    plt.figure()
    shap.summary_plot(shap_values[1], input_data, show=False)
    if not os.path.exists("static"):
        os.mkdir("static")
    plt.savefig("static/shap_summary.png", bbox_inches='tight')
    plt.close()
