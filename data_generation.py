import pandas as pd
import numpy as np

def generate_datasets():
    np.random.seed(42)

    material_types = [
        "Corrugated Cardboard", "Molded Pulp", "Biodegradable Plastic",
        "Recycled Paper", "Plant Fiber", "Starch-based Plastic"
    ]
    product_categories = ["Electronics", "Food", "Cosmetics", "Clothing", "Furniture"]

    # -----------------------------
    # Base Feature Generation
    # -----------------------------
    strength = np.random.uniform(5, 10, 80)
    weight_capacity = np.random.uniform(5, 25, 80)
    biodegradability_score = np.random.uniform(5, 10, 80)
    recyclability_percentage = np.random.uniform(50, 100, 80)

    # -----------------------------
    # LOGICAL TARGET GENERATION
    # -----------------------------

    # Cost increases with strength & capacity, decreases with recyclability
    cost_per_unit = (
        strength * 4 +
        weight_capacity * 1.2 -
        recyclability_percentage * 0.3 +
        np.random.normal(0, 5, 80)   # noise
    )

    # CO2 increases with weight & strength, decreases with biodegradability
    co2_emission_score = (
        weight_capacity * 0.25 +
        strength * 0.15 -
        biodegradability_score * 0.5 +
        np.random.normal(0, 1.5, 80)
    )

    # Clip values to keep realistic ranges
    cost_per_unit = np.clip(cost_per_unit, 10, 60)
    co2_emission_score = np.clip(co2_emission_score, 1, 10)

    materials_data = {
        "material_id": np.arange(1, 81),
        "material_type": np.random.choice(material_types, 80),
        "strength": np.round(strength, 2),
        "weight_capacity": np.round(weight_capacity, 2),
        "cost_per_unit": np.round(cost_per_unit, 2),
        "biodegradability_score": np.round(biodegradability_score, 2),
        "co2_emission_score": np.round(co2_emission_score, 2),
        "recyclability_percentage": np.round(recyclability_percentage, 2),
        "product_category": np.random.choice(product_categories, 80)
    }

    df_materials = pd.DataFrame(materials_data)

    # -----------------------------
    # Product dataset (same)
    # -----------------------------
    fragility_levels = ["Low", "Medium", "High"]
    shipping_types = ["Air", "Sea", "Road"]

    products_data = {
        "product_id": np.arange(1, 81),
        "product_name": ["Product_" + str(i) for i in range(1, 81)],
        "product_category": np.random.choice(product_categories, 80),
        "fragility_level": np.random.choice(fragility_levels, 80),
        "shipping_type": np.random.choice(shipping_types, 80)
    }

    df_products = pd.DataFrame(products_data)

    return df_materials, df_products

