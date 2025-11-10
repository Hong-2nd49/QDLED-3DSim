def suggest_mesh_settings(geometry) -> dict:
    """
    Placeholder for mesh / domain hints based on geometry.
    This is intentionally simple and can be expanded when integrating with COMSOL/TCAD.
    """
    total_thickness = sum(layer["thickness_nm"] for layer in geometry)
    return {
        "recommended_element_size_nm": max(1.0, total_thickness / 200.0),
        "notes": "Refine near QD/ETL interfaces and patterned regions."
    }
