from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


MOVE_GROUP_SEQUENCE_CAPABILITIES = (
    "pilz_industrial_motion_planner/MoveGroupSequenceAction "
    "pilz_industrial_motion_planner/MoveGroupSequenceService"
)


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("robotarm", package_name="myplan")
        .planning_pipelines(
            default_planning_pipeline="pilz_industrial_motion_planner",
            pipelines=["pilz_industrial_motion_planner", "ompl"],
        )
        .to_moveit_configs()
    )
    moveit_config.move_group_capabilities = {
        "capabilities": MOVE_GROUP_SEQUENCE_CAPABILITIES,
        "disable_capabilities": "",
    }
    return generate_move_group_launch(moveit_config)
