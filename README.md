# 🛒 Will This Visitor Buy? Purchase Intent Predictor

Most online store visitors leave without buying. This project predicts from a browsing session whether the visitor will purchase, so the store can show a coupon or a chat assistant only to the visitors who need it, instead of giving discounts to everyone.

  **👉 Live app:** [https://purchase-intent-predictor-gbahealvuy4fsxd97zfhhw.streamlit.app/

![App screenshot](screenshot.png)
<!-- Upload a screenshot of your app to the repo and name it screenshot.png -->

## What the app does
Enter a browsing session (pages visited, time spent, bounce and exit rates, page value, month, visitor type) and the app returns:
- the **probability that the visitor will buy**
- a **suggested action**: no action / show a coupon / offer the chat assistant (thresholds can be changed from the sidebar)

## Results
Dataset: 12,330 real sessions, about 15% of which end in a purchase. Models were tuned with 3-fold cross-validation on the training set only, then scored once on an untouched 20% test set (threshold 0.5).

| Model | CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline (always "no purchase") | - | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.899 | 0.911 | 0.512 | 0.793 | 0.622 |
| Random Forest | 0.925 | 0.930 | 0.590 | 0.777 | 0.671 |
| **Gradient Boosting (best)** | **0.930** | **0.937** | 0.719 | 0.615 | 0.663 |

**Choosing the threshold:** lowering it from 0.5 to 0.3 raises recall from 61.5% to 76.7% (precision 63.6%), so the store catches more real buyers at the cost of more false alarms.

**Key finding:** accuracy is misleading here (a model that always says "no purchase" is already about 84% accurate), so models were compared with ROC-AUC, precision and recall.

## Project workflow
1. Problem framing and metric choice
2. Data understanding and EDA
3. Preprocessing inside a scikit-learn `Pipeline` (split first, so there is no data leakage)
4. Baseline + Logistic Regression, Random Forest, Gradient Boosting
5. Cross-validation and hyperparameter tuning (`RandomizedSearchCV`)
6. Evaluation on the test set, threshold analysis, error analysis, feature importance
7. Business recommendations and limitations
8. Streamlit web app

## Limitations
- Data comes from a single store and period, so the model may not generalize to other stores or seasons.
- `PageValues` is computed from past behavior and must be available in real time in production.
- Buyers with no browsing history are often missed by the model.
- The model estimates purchase probability. It does not prove that a coupon causes a purchase; that needs an A/B test.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Files: `app.py` (web app), `model.joblib` (trained pipeline), `app_config.json` (default values and columns), `requirements.txt`.
The full analysis is in the notebook: `Purchase_Intent_Final_Project.ipynb`.

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit, joblib.

## Dataset
Sakar, C. and Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/468
