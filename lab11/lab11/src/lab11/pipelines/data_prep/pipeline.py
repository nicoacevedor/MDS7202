"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_data, preprocess_companies, preprocess_shuttles, create_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_data,
            inputs=None,
            outputs=["companies", "shuttles", "reviews"],
            name="get_data"
        ),
        node(
            func=preprocess_companies,
            inputs="companies",
            outputs="companies_prep",
            name="preprocess_companies"
        ),
        node(
            func=preprocess_shuttles,
            inputs="shuttles",
            outputs="shuttles_prep",
            name="preprocess_shuttles"
        ),
        node(
            func=create_model_input_table,
            inputs=["shuttles_prep", "companies_prep", "reviews"],
            outputs="prepared_data",
            name="create_model_input_table"
        )
    ])
