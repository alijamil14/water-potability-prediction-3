# Water Potability Streamlit deployment kit

Copy `app.py`, `requirements.txt`, and `.streamlit/config.toml` into your existing
`Water_Potability_Streamlit_Project` folder, replacing the old `app.py`.

Your project folder must contain:

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `data/water_potability.csv`
- `models/water_potability_model.pkl`
- `models/evaluation_metrics.json`

## Run locally

From the project folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Push the project folder contents to a GitHub repository.
2. Make sure the CSV, model `.pkl`, and metrics JSON are included in the repository.
3. Open https://share.streamlit.io and sign in with GitHub.
4. Select **Create app**.
5. Choose your repository and branch.
6. Set the main file path to `app.py` (if the project folder contents are at repository root).
7. Click **Deploy**.

Do not upload passwords, API keys, or other secrets to GitHub. Only load a Joblib model file that you created and trust.
