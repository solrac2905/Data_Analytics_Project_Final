from kedro.pipeline import Pipeline, node, pipeline
from .nodes import pre_processing_raw_data, apply_log_transform


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=pre_processing_raw_data,
                inputs=[
                    "scoring_data_raw",
                    "parameters",
                ],
                outputs="preprocessed_dataset",
                name="preprocessing_raw_node",
            ),
            node(
                func=apply_log_transform,
                inputs=["preprocessed_dataset"],
                outputs="preprocessed_log_dataset",
                name="apply_log_transform_node",
            ),
        ]
    )
