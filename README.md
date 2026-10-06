# Machine Learning Portfolio

Hands-on projects on **privacy-preserving machine learning and synthetic data**, presented in an interactive Streamlit app.

## Projects

| Notebook | Project |
|---|---|
| [`data.ipynb`](data.ipynb) | **Synthetic tabular data:** generates synthetic data from a real dataset with SDV and compares the two ([`real_data.csv`](real_data.csv) vs [`synthetic_data.csv`](synthetic_data.csv)) |
| [`project2.ipynb`](project2.ipynb) | **Privacy-preserving health diagnostics:** trains a PyTorch neural network on the Breast Cancer dataset with formal differential-privacy guarantees using Meta's Opacus |
| [`project3.ipynb`](project3.ipynb) | **Tabular diffusion model (TabDDPM) from scratch:** noise schedule, forward process, denoiser network, training loop and reverse-process sampling. Trained weights in `tabddpm_model.pth` |

## Interactive app

[`app.py`](app.py) is a Streamlit portfolio site that runs the models live.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python · PyTorch · Opacus · SDV · scikit-learn · pandas · Plotly · Streamlit
