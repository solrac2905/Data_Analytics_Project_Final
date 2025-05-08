"""Project pipelines."""

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """

#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines

from kedro.pipeline import Pipeline
from ambs_analytics_project.pipelines.data_processing import create_pipeline as data_processing
from ambs_analytics_project.pipelines.data_science import create_pipeline as data_science
 
def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": data_processing() + data_science(),
        "data_processing": data_processing(),
        "data_science": data_science(),
    }
