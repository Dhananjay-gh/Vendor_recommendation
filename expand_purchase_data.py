import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    print("Loading vendors...")
    df_v = pd.read_csv('processed_sap_vendors.csv')
    lifnrs = df_v['LIFNR'].tolist()

    # The user's original product catalog
    products = [
        {"name": "Enterprise Laptop", "range": (1100, 1600), "rate": 0.9},
        {"name": "Corporate Smartphone", "range": (600, 1200), "rate": 0.95},
        {"name": "Rack Server", "range": (8000, 12000), "rate": 0.85},
        {"name": "Cloud Compute Credit", "range": (100, 500), "rate": 1.05},
    ]

    start_date = datetime(2024, 5, 1)
    data = []
    np.random.seed(42)
    
    print(f"Generating data for {len(lifnrs)} vendors...")
    for v in lifnrs:
        vendor_bias = np.random.uniform(0.85, 1.15)
        
        for p in products:
            p_name = p["name"]
            low, high = p["range"]
            rate = p["rate"]
            
            num_transactions = np.random.randint(2, 5) # 2 to 4 transactions
            for _ in range(num_transactions):
                days_offset = np.random.randint(0, 730)
                date = start_date + timedelta(days=days_offset)
                years_passed = days_offset / 365.0
                multiplier = (rate ** years_passed)
                
                base_price = np.random.randint(low, high) * vendor_bias
                adjusted_price = round(base_price * multiplier, 2)
                
                data.append([p_name, adjusted_price, date.strftime("%Y-%m-%d"), v])

    df_price = pd.DataFrame(data, columns=["product_name", "price_per_unit", "date", "vendor_id"])
    df_price.to_csv('purchase_data.csv', index=False)
    print(f"Done! Restored original catalog and expanded dataset to {len(df_price)} rows.")

if __name__ == "__main__":
    main()
