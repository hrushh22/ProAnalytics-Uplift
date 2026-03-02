"""
========================================================================
PROMOTION UPLIFT MODELING USING CAUSAL MACHINE LEARNING
Team: ProAnalytics
========================================================================
Full Pipeline:
  1. Data Simulation (Dunnhumby-style retail data)
  2. Feature Engineering
  3. Treatment Assignment & Propensity Scoring
  4. Baseline Predictive Models
  5. T-Learner & X-Learner Causal Models
  6. Uplift Evaluation (Uplift Curve, Qini Curve)
  7. Customer Segmentation (K-Means + PCA)
  8. Export results to JSON for Dashboard
========================================================================
NOTE: If you have the actual Dunnhumby dataset, replace the
      simulate_dunnhumby_data() function with your real data loader.
      The rest of the pipeline will work as-is.
========================================================================
"""

import numpy as np
import pandas as pd
import json
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb

# Import real data loader if available
try:
    from data_loader import load_dunnhumby_data
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

# Import model saving utilities
try:
    from model_utils import save_pipeline_models
    MODEL_SAVE_AVAILABLE = True
except ImportError:
    MODEL_SAVE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# STEP 1: DATA SIMULATION (Replace with real Dunnhumby loader)
# ─────────────────────────────────────────────────────────────

def simulate_dunnhumby_data(n_customers=5000, seed=42):
    """
    Simulates Dunnhumby Complete Journey-style data.
    Mimics real retail customer behavior patterns.
    
    To use REAL data:
      df = pd.read_csv('transaction_data.csv')
      # Then map to the same column names used below
    """
    np.random.seed(seed)
    n = n_customers

    print(f"[DATA] Simulating {n} customers with Dunnhumby-style features...")

    # Customer demographics
    age_group = np.random.choice(['18-25','26-35','36-45','46-55','55+'], n,
                                  p=[0.12, 0.25, 0.28, 0.20, 0.15])
    income_group = np.random.choice(['Low','Medium','High'], n, p=[0.30, 0.45, 0.25])
    household_size = np.random.choice([1,2,3,4,5], n, p=[0.20,0.30,0.25,0.15,0.10])
    marital_status = np.random.choice(['Single','Married'], n, p=[0.40, 0.60])

    # Behavioral features (last 12 weeks)
    purchase_frequency = np.random.poisson(lam=3.5, size=n).clip(0, 15)
    avg_basket_value = np.random.lognormal(mean=3.5, sigma=0.6, size=n)
    recency_days = np.random.exponential(scale=14, size=n).clip(1, 90).astype(int)
    num_categories = np.random.randint(1, 12, size=n)
    coupon_redemption_rate = np.random.beta(2, 5, size=n)
    weekend_shopper = np.random.binomial(1, 0.45, size=n)
    store_visits = np.random.poisson(lam=4, size=n).clip(1, 20)

    # Historical promo response
    past_promo_response = np.random.beta(3, 7, size=n)
    discount_sensitivity = np.random.beta(2, 4, size=n)

    # Propensity to be treated (promotion targeting — realistic selection bias)
    # Higher frequency, higher basket value customers are more likely to be targeted
    propensity_raw = (
        0.3 * (purchase_frequency / 15) +
        0.2 * (avg_basket_value / avg_basket_value.max()) +
        0.15 * past_promo_response +
        0.10 * coupon_redemption_rate +
        0.05 * weekend_shopper +
        0.20 * np.random.random(n)  # noise
    )
    propensity = 1 / (1 + np.exp(-5 * (propensity_raw - 0.5)))  # sigmoid
    treatment = np.random.binomial(1, propensity)

    # Outcome: Purchase (with true causal uplift effect for some customers)
    # True uplift is heterogeneous — some customers respond, others don't
    true_uplift = (
        0.4 * discount_sensitivity +
        0.3 * coupon_redemption_rate +
        0.2 * (1 - recency_days / 90) +
        0.1 * past_promo_response
    )
    true_uplift = true_uplift / true_uplift.max()  # normalize to [0,1]

    base_purchase_prob = (
        0.25 +
        0.20 * (purchase_frequency / 15) +
        0.10 * (num_categories / 12) +
        0.15 * (avg_basket_value / avg_basket_value.max()) +
        0.05 * weekend_shopper
    ).clip(0.05, 0.90)

    # Treated outcome: base + causal treatment effect
    treated_prob = (base_purchase_prob + 0.25 * true_uplift * treatment).clip(0, 1)
    purchase = np.random.binomial(1, treated_prob)

    # Revenue outcome (conditional on purchase)
    revenue_base = avg_basket_value * purchase
    revenue_uplift = revenue_base * (1 + 0.15 * true_uplift * treatment)

    # Build dataframe
    df = pd.DataFrame({
        'customer_id': [f'C{str(i).zfill(5)}' for i in range(n)],
        'age_group': age_group,
        'income_group': income_group,
        'household_size': household_size,
        'marital_status': marital_status,
        'purchase_frequency': purchase_frequency,
        'avg_basket_value': avg_basket_value.round(2),
        'recency_days': recency_days,
        'num_categories': num_categories,
        'coupon_redemption_rate': coupon_redemption_rate.round(4),
        'weekend_shopper': weekend_shopper,
        'store_visits': store_visits,
        'past_promo_response': past_promo_response.round(4),
        'discount_sensitivity': discount_sensitivity.round(4),
        'propensity_score': propensity.round(4),
        'treatment': treatment,
        'purchase': purchase,
        'revenue': revenue_uplift.round(2),
        'true_uplift': true_uplift.round(4),  # ground truth (available only in simulation)
    })

    n_treated = treatment.sum()
    n_control = n - n_treated
    print(f"[DATA] Treatment group: {n_treated} ({n_treated/n*100:.1f}%)")
    print(f"[DATA] Control group:   {n_control} ({n_control/n*100:.1f}%)")
    print(f"[DATA] Overall purchase rate: {purchase.mean()*100:.1f}%")
    print(f"[DATA] Treated purchase rate: {purchase[treatment==1].mean()*100:.1f}%")
    print(f"[DATA] Control purchase rate: {purchase[treatment==0].mean()*100:.1f}%")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def feature_engineering(df):
    """Creates model-ready features from raw data."""
    print("\n[FEATURES] Engineering features...")

    fe = df.copy()

    # Encode categorical variables
    fe['age_encoded'] = fe['age_group'].map(
        {'18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '55+': 5})
    fe['income_encoded'] = fe['income_group'].map({'Low': 1, 'Medium': 2, 'High': 3})
    fe['married'] = (fe['marital_status'] == 'Married').astype(int)
    
    # Handle household_size - convert to numeric if it's a string
    if fe['household_size'].dtype == 'object':
        # Extract number from strings like "2 Adults No Kids" or just use hash
        fe['household_size'] = pd.factorize(fe['household_size'])[0] + 1
    
    # Fill NaN values in encoded columns
    fe['age_encoded'] = fe['age_encoded'].fillna(3)  # default to middle age
    fe['income_encoded'] = fe['income_encoded'].fillna(2)  # default to medium

    # Interaction features
    fe['promo_sensitivity_score'] = (
        fe['coupon_redemption_rate'] * 0.5 +
        fe['past_promo_response'] * 0.3 +
        fe['discount_sensitivity'] * 0.2
    )
    fe['engagement_score'] = (
        fe['purchase_frequency'] / fe['purchase_frequency'].max() * 0.4 +
        fe['store_visits'] / fe['store_visits'].max() * 0.3 +
        fe['num_categories'] / fe['num_categories'].max() * 0.3
    )
    fe['value_score'] = (
        fe['avg_basket_value'] / fe['avg_basket_value'].max() * 0.6 +
        fe['income_encoded'] / 3 * 0.4
    )
    fe['recency_score'] = 1 - (fe['recency_days'] / 90)
    fe['rfm_score'] = (
        fe['recency_score'] * 0.3 +
        fe['purchase_frequency'] / fe['purchase_frequency'].max() * 0.4 +
        fe['value_score'] * 0.3
    )

    print(f"[FEATURES] Total features created: {len(get_feature_cols(fe))}")
    return fe


def get_feature_cols(df):
    """Returns list of feature columns for modeling."""
    return [
        'age_encoded', 'income_encoded', 'household_size', 'married',
        'purchase_frequency', 'avg_basket_value', 'recency_days',
        'num_categories', 'coupon_redemption_rate', 'weekend_shopper',
        'store_visits', 'past_promo_response', 'discount_sensitivity',
        'promo_sensitivity_score', 'engagement_score', 'value_score',
        'recency_score', 'rfm_score'
    ]


# ─────────────────────────────────────────────────────────────
# STEP 3: PROPENSITY SCORE MODELING
# ─────────────────────────────────────────────────────────────

def fit_propensity_model(df, feature_cols):
    """
    Estimate P(T=1|X) to check overlap and correct for selection bias.
    """
    print("\n[PROPENSITY] Fitting propensity score model...")

    X = df[feature_cols].values
    T = df['treatment'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ps_model = LogisticRegression(max_iter=500, C=1.0)
    ps_model.fit(X_scaled, T)

    ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
    ps_auc = roc_auc_score(T, ps_scores)

    print(f"[PROPENSITY] Propensity model AUC: {ps_auc:.4f}")
    print(f"[PROPENSITY] Propensity score range: [{ps_scores.min():.3f}, {ps_scores.max():.3f}]")
    print(f"[PROPENSITY] Mean propensity (treated): {ps_scores[T==1].mean():.3f}")
    print(f"[PROPENSITY] Mean propensity (control): {ps_scores[T==0].mean():.3f}")

    return ps_model, scaler, ps_scores


# ─────────────────────────────────────────────────────────────
# STEP 4: BASELINE MODELS
# ─────────────────────────────────────────────────────────────

def fit_baseline_models(df, feature_cols):
    """
    Baseline predictive models — predict P(purchase) ignoring causality.
    """
    print("\n[BASELINE] Training baseline predictive models...")

    X = df[feature_cols].values
    y = df['purchase'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42,
                                      eval_metric='logloss', verbosity=0),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            preds = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, preds)
        results[name] = {'auc': round(auc, 4), 'model': model}
        print(f"[BASELINE] {name}: AUC = {auc:.4f}")

    # Feature importance from XGBoost
    xgb_model = results['XGBoost']['model']
    feature_imp = dict(zip(feature_cols,
                           xgb_model.feature_importances_.tolist()))
    feature_imp = dict(sorted(feature_imp.items(),
                               key=lambda x: x[1], reverse=True))

    print(f"\n[BASELINE] Top 5 features:")
    for feat, imp in list(feature_imp.items())[:5]:
        print(f"  {feat}: {imp:.4f}")

    return results, feature_imp, scaler


# ─────────────────────────────────────────────────────────────
# STEP 5: T-LEARNER (Causal Uplift Model)
# ─────────────────────────────────────────────────────────────

def fit_t_learner(df, feature_cols):
    """
    T-Learner: Train separate models on treated and control groups.
    Uplift = P(Y=1|T=1, X) - P(Y=1|T=0, X)
    """
    print("\n[T-LEARNER] Fitting T-Learner causal model...")

    X = df[feature_cols].values
    T = df['treatment'].values
    Y = df['purchase'].values

    X_treat = X[T == 1]
    Y_treat = Y[T == 1]
    X_control = X[T == 0]
    Y_control = Y[T == 0]

    # Model for treated group
    m1 = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                             random_state=42, eval_metric='logloss', verbosity=0)
    m1.fit(X_treat, Y_treat)

    # Model for control group
    m0 = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                             random_state=42, eval_metric='logloss', verbosity=0)
    m0.fit(X_control, Y_control)

    # Predict on all customers
    p_treat = m1.predict_proba(X)[:, 1]
    p_control = m0.predict_proba(X)[:, 1]
    uplift_t = p_treat - p_control

    # Evaluate against true uplift (simulation only)
    corr = np.corrcoef(uplift_t, df['true_uplift'].values)[0, 1]
    print(f"[T-LEARNER] Correlation with true uplift: {corr:.4f}")
    print(f"[T-LEARNER] Mean predicted uplift: {uplift_t.mean():.4f}")
    print(f"[T-LEARNER] Uplift range: [{uplift_t.min():.4f}, {uplift_t.max():.4f}]")

    return m1, m0, uplift_t, p_treat, p_control


# ─────────────────────────────────────────────────────────────
# STEP 6: X-LEARNER (Improved Causal Uplift Model)
# ─────────────────────────────────────────────────────────────

def fit_x_learner(df, feature_cols, ps_scores):
    """
    X-Learner: Better handling of imbalanced treatment/control.
    Uses imputed treatment effects and propensity-weighted combination.
    """
    print("\n[X-LEARNER] Fitting X-Learner causal model...")

    X = df[feature_cols].values
    T = df['treatment'].values
    Y = df['purchase'].values

    X_treat = X[T == 1]
    Y_treat = Y[T == 1]
    X_control = X[T == 0]
    Y_control = Y[T == 0]
    ps_treat = ps_scores[T == 1]
    ps_control = ps_scores[T == 0]

    # Stage 1: Fit base models
    m1 = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                             random_state=42, eval_metric='logloss', verbosity=0)
    m1.fit(X_treat, Y_treat)

    m0 = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                             random_state=42, eval_metric='logloss', verbosity=0)
    m0.fit(X_control, Y_control)

    # Stage 2: Imputed treatment effects
    d1 = Y_treat - m0.predict_proba(X_treat)[:, 1]   # treated group: actual - predicted control
    d0 = m1.predict_proba(X_control)[:, 1] - Y_control  # control group: predicted treated - actual

    # Stage 3: Fit treatment effect models
    tau1 = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                              random_state=42, verbosity=0)
    tau1.fit(X_treat, d1)

    tau0 = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                              random_state=42, verbosity=0)
    tau0.fit(X_control, d0)

    # Stage 4: Propensity-weighted combination on ALL customers
    tau1_all = tau1.predict(X)
    tau0_all = tau0.predict(X)
    g = ps_scores  # propensity scores for all
    uplift_x = g * tau0_all + (1 - g) * tau1_all

    corr = np.corrcoef(uplift_x, df['true_uplift'].values)[0, 1]
    print(f"[X-LEARNER] Correlation with true uplift: {corr:.4f}")
    print(f"[X-LEARNER] Mean predicted uplift: {uplift_x.mean():.4f}")
    print(f"[X-LEARNER] Uplift range: [{uplift_x.min():.4f}, {uplift_x.max():.4f}]")

    return uplift_x


# ─────────────────────────────────────────────────────────────
# STEP 7: UPLIFT EVALUATION
# ─────────────────────────────────────────────────────────────

def compute_uplift_curve(T, Y, uplift_scores, n_bins=20):
    """
    Compute uplift curve and Qini curve data.
    Returns percentile-level cumulative uplift metrics.
    """
    df_eval = pd.DataFrame({
        'treatment': T,
        'purchase': Y,
        'uplift': uplift_scores
    }).sort_values('uplift', ascending=False).reset_index(drop=True)

    n = len(df_eval)
    bin_size = n // n_bins

    uplift_curve = []
    qini_curve = []
    random_curve = []
    cumulative_treated = 0
    cumulative_control = 0
    cumulative_treated_buyers = 0
    cumulative_control_buyers = 0

    total_treated = T.sum()
    total_control = (1 - T).sum()
    total_buyers_treated = Y[T == 1].sum()

    for i in range(n_bins):
        chunk = df_eval.iloc[i * bin_size:(i + 1) * bin_size]
        cumulative_treated += chunk['treatment'].sum()
        cumulative_control += (1 - chunk['treatment']).sum()
        cumulative_treated_buyers += chunk[chunk['treatment'] == 1]['purchase'].sum()
        cumulative_control_buyers += chunk[chunk['treatment'] == 0]['purchase'].sum()

        pct_targeted = (i + 1) / n_bins * 100

        # Uplift curve: (treated_cr - control_cr) * N_targeted
        if cumulative_treated > 0 and cumulative_control > 0:
            treated_cr = cumulative_treated_buyers / cumulative_treated
            control_cr = cumulative_control_buyers / cumulative_control
            uplift_val = (treated_cr - control_cr) * (i + 1) * bin_size / n * 100
        else:
            uplift_val = 0

        # Qini: cumulative treated buyers - cumulative_treated / N * total_treated_buyers
        qini_val = cumulative_treated_buyers - (cumulative_treated / total_treated) * total_buyers_treated if total_treated > 0 else 0
        random_val = (i + 1) / n_bins * 100

        uplift_curve.append({'percentile': round(pct_targeted, 1), 'model': round(uplift_val, 3)})
        qini_curve.append({'percentile': round(pct_targeted, 1),
                           'model': round(qini_val, 2), 'random': 0})
        random_curve.append(round(random_val, 1))

    # Qini coefficient (area between model and random)
    model_qini = [q['model'] for q in qini_curve]
    qini_coeff = np.trapz(model_qini) / n_bins
    print(f"[EVAL] Qini coefficient: {qini_coeff:.4f}")

    return uplift_curve, qini_curve, qini_coeff


# ─────────────────────────────────────────────────────────────
# STEP 8: CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────────────────────

def segment_customers(df, uplift_scores):
    """
    Segment customers into 4 uplift groups + K-Means behavioral clusters.
    """
    print("\n[SEGMENT] Segmenting customers...")

    # Uplift-based segments
    def assign_uplift_segment(score):
        if score > 0.15:
            return 'Persuadable'
        elif score > 0.05:
            return 'Moderate Responder'
        elif score > -0.05:
            return 'Sure Thing / Neutral'
        else:
            return 'Do-Not-Disturb'

    segments = [assign_uplift_segment(s) for s in uplift_scores]
    segment_counts = pd.Series(segments).value_counts().to_dict()

    print("[SEGMENT] Uplift segments:")
    for seg, cnt in segment_counts.items():
        print(f"  {seg}: {cnt} ({cnt/len(segments)*100:.1f}%)")

    # K-Means clustering on behavioral features
    cluster_features = ['rfm_score', 'promo_sensitivity_score',
                        'engagement_score', 'value_score', 'recency_score']
    X_cluster = df[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Sample 500 points for dashboard visualization
    sample_idx = np.random.choice(len(df), size=min(500, len(df)), replace=False)
    pca_sample = [
        {
            'x': round(float(X_pca[i, 0]), 3),
            'y': round(float(X_pca[i, 1]), 3),
            'cluster': int(cluster_labels[i]),
            'segment': segments[i],
            'uplift': round(float(uplift_scores[i]), 4)
        }
        for i in sample_idx
    ]

    return segments, segment_counts, cluster_labels, pca_sample


# ─────────────────────────────────────────────────────────────
# STEP 9: ROI SIMULATION
# ─────────────────────────────────────────────────────────────

def simulate_roi(df, uplift_scores, segments, discount_cost_per_customer=5.0):
    """
    Simulate ROI at different targeting thresholds.
    """
    print("\n[ROI] Running ROI simulation...")

    df_sim = df.copy()
    df_sim['uplift_score'] = uplift_scores
    df_sim['segment'] = segments
    df_sim_sorted = df_sim.sort_values('uplift_score', ascending=False)

    roi_data = []
    total_customers = len(df_sim)

    for pct in range(5, 105, 5):
        n_target = int(total_customers * pct / 100)
        targeted = df_sim_sorted.iloc[:n_target]
        persuadables = targeted[targeted['segment'] == 'Persuadable']

        # Revenue from persuadables who would respond
        incremental_revenue = persuadables['revenue'].sum() * 0.25  # 25% uplift revenue
        total_discount_cost = n_target * discount_cost_per_customer
        net_roi = incremental_revenue - total_discount_cost

        roi_data.append({
            'pct_targeted': pct,
            'n_customers': n_target,
            'incremental_revenue': round(float(incremental_revenue), 2),
            'discount_cost': round(float(total_discount_cost), 2),
            'net_roi': round(float(net_roi), 2),
            'roi_pct': round(float(net_roi / total_discount_cost * 100) if total_discount_cost > 0 else 0, 1)
        })

    # Find optimal targeting threshold
    best = max(roi_data, key=lambda x: x['net_roi'])
    print(f"[ROI] Optimal targeting: Top {best['pct_targeted']}% of customers")
    print(f"[ROI] Max net ROI: ${best['net_roi']:,.0f}")

    return roi_data, best


# ─────────────────────────────────────────────────────────────
# STEP 10: EXPORT TO JSON FOR DASHBOARD
# ─────────────────────────────────────────────────────────────

def export_dashboard_data(df, uplift_t, uplift_x, p_treat, p_control,
                           segments, segment_counts, cluster_labels, pca_sample,
                           feature_imp, baseline_results, uplift_curve, qini_curve,
                           qini_coeff, roi_data, best_roi, ps_scores):
    """
    Exports all results to a JSON file that the dashboard reads.
    """
    print("\n[EXPORT] Exporting results for dashboard...")

    # Use X-Learner as primary uplift score (better model)
    df['uplift_x'] = uplift_x
    df['uplift_t'] = uplift_t
    df['p_treated'] = p_treat
    df['p_control'] = p_control
    df['segment'] = segments
    df['cluster'] = cluster_labels
    df['propensity'] = ps_scores

    # Summary KPIs
    n_total = len(df)
    n_persuadable = segment_counts.get('Persuadable', 0)
    n_sure_thing = segment_counts.get('Sure Thing / Neutral', 0)
    n_dnd = segment_counts.get('Do-Not-Disturb', 0)
    n_moderate = segment_counts.get('Moderate Responder', 0)
    avg_uplift = float(uplift_x.mean())
    treatment_cr = float(df[df['treatment']==1]['purchase'].mean())
    control_cr = float(df[df['treatment']==0]['purchase'].mean())

    # Uplift score distribution
    hist_counts, hist_bins = np.histogram(uplift_x, bins=30)
    uplift_histogram = [
        {'bin': round(float(hist_bins[i]), 3), 'count': int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    # Top 20 customers sample for table
    top_customers = df.nlargest(200, 'uplift_x')[
        ['customer_id', 'uplift_x', 'uplift_t', 'p_treated', 'p_control',
         'segment', 'purchase_frequency', 'avg_basket_value', 'rfm_score',
         'promo_sensitivity_score', 'treatment', 'purchase']
    ].round(4).to_dict('records')

    # Baseline model comparison
    model_comparison = {
        name: {'auc': vals['auc']}
        for name, vals in baseline_results.items()
    }

    # Treatment/control comparison
    treat_ctrl_comparison = {
        'treatment_purchase_rate': round(treatment_cr * 100, 2),
        'control_purchase_rate': round(control_cr * 100, 2),
        'avg_revenue_treated': round(float(df[df['treatment']==1]['revenue'].mean()), 2),
        'avg_revenue_control': round(float(df[df['treatment']==0]['revenue'].mean()), 2),
        'n_treated': int(df['treatment'].sum()),
        'n_control': int((df['treatment']==0).sum()),
    }

    # Weekly trend simulation for sparklines
    np.random.seed(99)
    weekly_trend = [round(float(x), 2) for x in
                    np.cumsum(np.random.randn(12) * 0.5 + 2) + 50]

    # Build final JSON
    dashboard_data = {
        'meta': {
            'project': 'Promotion Uplift Modeling Using Causal ML',
            'team': 'ProAnalytics',
            'dataset': 'Dunnhumby Complete Journey (Simulated)',
            'n_customers': n_total,
            'models_used': ['T-Learner (XGBoost)', 'X-Learner (XGBoost)', 'Propensity Score (LR)']
        },
        'kpis': {
            'total_customers': n_total,
            'persuadable_count': n_persuadable,
            'persuadable_pct': round(n_persuadable / n_total * 100, 1),
            'avg_uplift_score': round(avg_uplift, 4),
            'treatment_purchase_rate': round(treatment_cr * 100, 2),
            'control_purchase_rate': round(control_cr * 100, 2),
            'lift': round((treatment_cr - control_cr) * 100, 2),
            'qini_coefficient': round(qini_coeff, 4),
            'optimal_targeting_pct': best_roi['pct_targeted'],
            'max_net_roi': best_roi['net_roi'],
            'total_revenue': round(float(df['revenue'].sum()), 0),
        },
        'segments': {
            'Persuadable': n_persuadable,
            'Moderate Responder': n_moderate,
            'Sure Thing / Neutral': n_sure_thing,
            'Do-Not-Disturb': n_dnd,
        },
        'uplift_histogram': uplift_histogram,
        'uplift_curve': uplift_curve,
        'qini_curve': qini_curve,
        'feature_importance': [
            {'feature': k.replace('_', ' ').title(), 'importance': round(v, 4)}
            for k, v in list(feature_imp.items())[:12]
        ],
        'baseline_models': model_comparison,
        'treatment_control': treat_ctrl_comparison,
        'roi_simulation': roi_data,
        'best_roi': best_roi,
        'pca_clusters': pca_sample,
        'top_customers': top_customers[:100],  # top 100 for table
        'weekly_trend': weekly_trend,
    }

    # Save to outputs folder
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"[EXPORT] Data exported to outputs/dashboard_data.json")
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"  Total customers analyzed: {n_total:,}")
    print(f"  Persuadable customers:    {n_persuadable:,} ({n_persuadable/n_total*100:.1f}%)")
    print(f"  Average uplift score:     {avg_uplift:.4f}")
    print(f"  Qini coefficient:         {qini_coeff:.4f}")
    print(f"  Optimal targeting:        Top {best_roi['pct_targeted']}%")
    print(f"  Max net ROI:              ${best_roi['net_roi']:,.0f}")
    print(f"{'='*60}")

    return dashboard_data


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("="*60)
    print("  PROMOTION UPLIFT MODELING - CAUSAL ML PIPELINE")
    print("  Team ProAnalytics")
    print("="*60)

    # 1. Load / simulate data
    # Check if real data is available
    if REAL_DATA_AVAILABLE and Path('data/transaction_data.csv').exists():
        print("\n[INFO] Real Dunnhumby data detected!")
        df = load_dunnhumby_data(sample_size=5000)
    else:
        print("\n[INFO] Using simulated data (real data not found)")
        print("[INFO] To use real data: run 'python download_data.py' first")
        df = simulate_dunnhumby_data(n_customers=5000)

    # 2. Feature engineering
    df = feature_engineering(df)
    feature_cols = get_feature_cols(df)

    # 3. Propensity score modeling
    ps_model, ps_scaler, ps_scores = fit_propensity_model(df, feature_cols)

    # 4. Baseline models
    baseline_results, feature_imp, _ = fit_baseline_models(df, feature_cols)

    # 5. T-Learner
    m1, m0, uplift_t, p_treat, p_control = fit_t_learner(df, feature_cols)

    # 6. X-Learner (primary model)
    uplift_x = fit_x_learner(df, feature_cols, ps_scores)

    # 7. Evaluation
    uplift_curve, qini_curve, qini_coeff = compute_uplift_curve(
        df['treatment'].values, df['purchase'].values, uplift_x)

    # 8. Segmentation
    segments, segment_counts, cluster_labels, pca_sample = segment_customers(df, uplift_x)

    # 9. ROI Simulation
    roi_data, best_roi = simulate_roi(df, uplift_x, segments)

    # 10. Export
    dashboard_data = export_dashboard_data(
        df, uplift_t, uplift_x, p_treat, p_control,
        segments, segment_counts, cluster_labels, pca_sample,
        feature_imp, baseline_results, uplift_curve, qini_curve,
        qini_coeff, roi_data, best_roi, ps_scores
    )

    # 11. Save models
    if MODEL_SAVE_AVAILABLE:
        print("\n[SAVE] Saving trained models...")
        models_to_save = {
            't_learner_treated': m1,
            't_learner_control': m0,
            'propensity_model': ps_model,
            'xgboost_baseline': baseline_results['XGBoost']['model']
        }
        save_pipeline_models(models_to_save)
    
    print("\nNext step: Open outputs/dashboard.html in your browser!")
