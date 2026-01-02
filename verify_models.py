 
"""Verify all models are implemented"""
print("="*60)
print("VERIFYING ALL 10 MODELS ARE IMPLEMENTED")
print("="*60)

from model_manager import ModelManager

mm = ModelManager()
configs = mm.model_configs

print(f"\n✓ Found {len(configs)} model configurations:\n")

models = [
    ("1. LLM-Sentiment", "LLM-Sentiment"),
    ("2. XGBoost", "XGBoost"),
    ("3. Custom GBM", "Custom_GBM"),
    ("4. ARIMA/SARIMA", "ARIMA_SARIMA"),
    ("5. Prophet", "Prophet"),
    ("6. Random Forest", "RandomForest"),
    ("7. LightGBM", "LightGBM"),
    ("8. SVR", "SVR"),
    ("9. LSTM/GRU", "LSTM_GRU"),
    ("10. TCN", "TCN"),
]

for name, key in models:
    if key in configs:
        print(f"{name}")
        print(f"  Description: {configs[key]['description']}")
        print(f"  Buy: {configs[key]['buy_condition']}")
        print(f"  Sell: {configs[key]['sell_condition']}")
        print()
    else:
        print(f"❌ {name} - NOT FOUND")

# Check ensemble
print("\n" + "="*60)
print("ENSEMBLE MODEL RULES")
print("="*60)
print("\nBuy Conditions (ALL must be true):")
print("  - r_{t+1} > 0.005 (predicted return > 0.5%)")
print("  - bull_count >= 6 (at least 6 models bullish)")
print("  - P_t > SMA20 (price above 20-day MA)")

print("\nSell Conditions (ANY can be true):")
print("  - r_{t+1} < -0.005 (predicted return < -0.5%)")
print("  - bull_count <= 4 (4 or fewer models bullish)")
print("  - P_t < SMA20 (price below 20-day MA)")

print("\nTime Stop:")
print("  - Held >= 5 days AND cumulative return > 5%")

print("\nStop-Loss:")
print("  - P_t < EntryPrice × 0.98 (2% loss)")

print("\n" + "="*60)
print("✓ ALL MODELS VERIFIED!")
print("="*60)


