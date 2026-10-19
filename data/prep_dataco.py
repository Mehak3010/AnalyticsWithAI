import pandas as pd


def prep_dataco(input_path: str, output_path: str):
    print(f"Loading data from {input_path}...")
    # Use latin-1 encoding as specified
    df = pd.read_csv(input_path, encoding="latin-1")

    # 1. Drop personal and unused columns
    cols_to_drop = [
        "Customer Fname",
        "Customer Lname",
        "Customer Email",
        "Customer Password",
        "Customer Street",
        "Product Image",
    ]
    df = df.drop(columns=cols_to_drop)

    # 2. Parse order date
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"])

    # 3. Derive metrics
    df["late_flag"] = (df["Delivery Status"] == "Late delivery").astype(int)
    df["gross_sales"] = df["Order Item Product Price"] * df["Order Item Quantity"]
    df["discount_amount"] = df["Order Item Discount"]

    # Preserve volume and unit price so the downstream variance bridge can
    # decompose revenue changes without trying to infer these inputs later.
    df["quantity"] = pd.to_numeric(df["Order Item Quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["Order Item Product Price"], errors="coerce")

    # 4. Select and rename columns for the tidy dataset
    column_mapping = {
        "Market": "market",
        "Order Region": "order_region",
        "Category Name": "category_name",
        "Shipping Mode": "shipping_mode",
        "Customer Segment": "customer_segment",
        "Sales": "sales",
        "Order Profit Per Order": "profit",
        "Order Item Id": "order_item_id",
    }

    available_cols = [c for c in column_mapping if c in df.columns]
    df = df.rename(columns={c: column_mapping[c] for c in available_cols})

    keep_cols = [
        "order_date",
        "market",
        "order_region",
        "category_name",
        "shipping_mode",
        "customer_segment",
        "sales",
        "profit",
        "order_item_id",
        "late_flag",
        "gross_sales",
        "discount_amount",
        "quantity",
        "unit_price",
    ]

    df = df[keep_cols]
    df.to_csv(output_path, index=False)
    print(f"Tidy data saved to {output_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    input_file = "data/DataCoSupplyChainDataset.csv"
    output_file = "data/dataco_tidy.csv"
    prep_dataco(input_file, output_file)
