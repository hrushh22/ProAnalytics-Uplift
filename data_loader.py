"""
Real Dunnhumby Data Loader
Loads and processes actual Dunnhumby Complete Journey dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_dunnhumby_data(data_dir='data', sample_size=None):
    """
    Load real Dunnhumby Complete Journey dataset
    
    Args:
        data_dir: Directory containing CSV files
        sample_size: Number of households to sample (None = all)
    
    Returns:
        DataFrame with customer features, treatment, and purchase outcome
    """
    print(f"[DATA] Loading Dunnhumby Complete Journey dataset from {data_dir}/")
    
    # Load all tables
    transactions = pd.read_csv(f'{data_dir}/transaction_data.csv')
    demographics = pd.read_csv(f'{data_dir}/hh_demographic.csv')
    campaigns = pd.read_csv(f'{data_dir}/campaign_table.csv')
    coupon_redempt = pd.read_csv(f'{data_dir}/coupon_redempt.csv')
    
    print(f"[DATA] Loaded {len(transactions):,} transactions from {transactions['household_key'].nunique():,} households")
    
    # Sample households if requested
    if sample_size:
        households = transactions['household_key'].unique()
        sampled_hh = np.random.choice(households, size=min(sample_size, len(households)), replace=False)
        transactions = transactions[transactions['household_key'].isin(sampled_hh)]
        print(f"[DATA] Sampled {sample_size:,} households")
    
    # Build customer-level features
    print("[DATA] Building customer features...")
    
    # 1. Transaction-based features (RFM + behavior)
    trans_features = transactions.groupby('household_key').agg({
        'BASKET_ID': 'nunique',  # purchase frequency
        'SALES_VALUE': ['sum', 'mean'],  # total spend, avg basket
        'QUANTITY': 'sum',
        'STORE_ID': 'nunique',
        'DAY': ['min', 'max']
    }).reset_index()
    
    trans_features.columns = ['household_key', 'purchase_frequency', 'total_spend', 
                               'avg_basket_value', 'total_quantity', 'num_stores', 
                               'first_day', 'last_day']
    
    trans_features['recency_days'] = trans_features['last_day'].max() - trans_features['last_day']
    trans_features['customer_tenure'] = trans_features['last_day'] - trans_features['first_day']
    
    # Category diversity
    cat_diversity = transactions.groupby('household_key')['PRODUCT_ID'].nunique().reset_index()
    cat_diversity.columns = ['household_key', 'num_categories']
    trans_features = trans_features.merge(cat_diversity, on='household_key')
    
    # Weekend shopping
    weekend_shop = transactions.copy()
    weekend_shop['is_weekend'] = weekend_shop['DAY'] % 7 >= 5
    weekend_pct = weekend_shop.groupby('household_key')['is_weekend'].mean().reset_index()
    weekend_pct.columns = ['household_key', 'weekend_shopper']
    trans_features = trans_features.merge(weekend_pct, on='household_key')
    
    # 2. Coupon redemption behavior
    coupon_features = coupon_redempt.groupby('household_key').agg({
        'COUPON_UPC': 'count'
    }).reset_index()
    coupon_features.columns = ['household_key', 'coupons_redeemed']
    
    # Merge with all households
    trans_features = trans_features.merge(coupon_features, on='household_key', how='left')
    trans_features['coupons_redeemed'] = trans_features['coupons_redeemed'].fillna(0)
    trans_features['coupon_redemption_rate'] = (trans_features['coupons_redeemed'] / 
                                                 trans_features['purchase_frequency']).clip(0, 1)
    
    # 3. Campaign exposure (treatment assignment)
    # DESCRIPTION column is already in campaigns table
    treatment_campaigns = campaigns[campaigns['DESCRIPTION'] == 'TypeA']
    treated_hh = treatment_campaigns['household_key'].unique()
    
    trans_features['treatment'] = trans_features['household_key'].isin(treated_hh).astype(int)
    
    # 4. Create synthetic past_promo_response (not directly available)
    np.random.seed(42)
    trans_features['past_promo_response'] = np.random.beta(3, 7, len(trans_features))
    
    # 5. Demographics
    demographics_clean = demographics.copy()
    demographics_clean.columns = demographics_clean.columns.str.lower()
    trans_features = trans_features.merge(demographics_clean, 
                                          left_on='household_key', 
                                          right_on='household_key', 
                                          how='left')
    
    # 6. Outcome: Purchase in treatment period
    # Define treatment period as last 30 days
    treatment_period = transactions['DAY'].max() - 30
    recent_purchases = transactions[transactions['DAY'] >= treatment_period]
    purchasers = recent_purchases.groupby('household_key').agg({
        'SALES_VALUE': 'sum'
    }).reset_index()
    purchasers.columns = ['household_key', 'revenue']
    purchasers['purchase'] = 1
    
    trans_features = trans_features.merge(purchasers[['household_key', 'purchase', 'revenue']], 
                                          on='household_key', how='left')
    trans_features['purchase'] = trans_features['purchase'].fillna(0).astype(int)
    trans_features['revenue'] = trans_features['revenue'].fillna(0)
    
    # 7. Create synthetic discount sensitivity (not in dataset)
    np.random.seed(42)
    trans_features['discount_sensitivity'] = np.random.beta(2, 4, len(trans_features))
    
    # 8. Propensity score (probability of being treated)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = ['purchase_frequency', 'avg_basket_value', 'total_spend', 
                    'num_categories', 'coupon_redemption_rate']
    X_prop = trans_features[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_prop)
    
    ps_model = LogisticRegression(max_iter=500)
    ps_model.fit(X_scaled, trans_features['treatment'])
    trans_features['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]
    
    # Rename for consistency with pipeline
    df = trans_features.rename(columns={
        'household_key': 'customer_id',
        'age_desc': 'age_group',
        'income_desc': 'income_group',
        'homeowner_desc': 'marital_status',
        'hh_comp_desc': 'household_size',
        'num_stores': 'store_visits'
    })
    
    # Clean up columns
    df['age_group'] = df['age_group'].fillna('Unknown')
    df['income_group'] = df['income_group'].fillna('Unknown')
    df['marital_status'] = df['marital_status'].fillna('Unknown')
    df['household_size'] = df['household_size'].fillna('Unknown')
    
    # Add true_uplift placeholder (unknown in real data)
    df['true_uplift'] = np.nan
    
    n_treated = df['treatment'].sum()
    n_control = len(df) - n_treated
    print(f"[DATA] Treatment group: {n_treated} ({n_treated/len(df)*100:.1f}%)")
    print(f"[DATA] Control group:   {n_control} ({n_control/len(df)*100:.1f}%)")
    print(f"[DATA] Overall purchase rate: {df['purchase'].mean()*100:.1f}%")
    print(f"[DATA] Treated purchase rate: {df[df['treatment']==1]['purchase'].mean()*100:.1f}%")
    print(f"[DATA] Control purchase rate: {df[df['treatment']==0]['purchase'].mean()*100:.1f}%")
    
    return df
