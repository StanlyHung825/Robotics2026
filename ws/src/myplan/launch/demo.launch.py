from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("robotarm", package_name="myplan")
        .planning_pipelines(
            default_planning_pipeline="pilz_industrial_motion_planner",
            pipelines=["pilz_industrial_motion_planner", "ompl"],
        )
        .to_moveit_configs()
    )
    return generate_demo_launch(moveit_config)
