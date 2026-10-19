import pandas as pd

from data.prep_dataco import prep_dataco


def test_prep_dataco_preserves_quantity_and_unit_price(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "tidy.csv"
    pd.DataFrame(
        {
            "Customer Fname": ["A"],
            "Customer Lname": ["B"],
            "Customer Email": ["a@example.com"],
            "Customer Password": ["unused"],
            "Customer Street": ["unused"],
            "Product Image": ["unused"],
            "order date (DateOrders)": ["2017-01-01"],
            "Delivery Status": ["Late delivery"],
            "Order Item Product Price": [12.5],
            "Order Item Quantity": [3],
            "Order Item Discount": [1.5],
            "Market": ["Europe"],
            "Order Region": ["Western Europe"],
            "Category Name": ["Books"],
            "Shipping Mode": ["Standard Class"],
            "Customer Segment": ["Consumer"],
            "Sales": [36.0],
            "Order Profit Per Order": [4.0],
            "Order Item Id": [101],
        }
    ).to_csv(source, index=False)

    prep_dataco(str(source), str(output))
    result = pd.read_csv(output)

    assert result.loc[0, "quantity"] == 3
    assert result.loc[0, "unit_price"] == 12.5
    assert result.loc[0, "gross_sales"] == 37.5
    assert result.loc[0, "sales"] == 36.0
