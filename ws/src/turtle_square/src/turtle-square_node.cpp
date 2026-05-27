#include "turtle_square/turtle-square_node.hpp"
#include <cmath>

TurtleSquareNode::TurtleSquareNode() : Node("turtle_square_node"){
    _cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10);
    _timer = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&TurtleSquareNode::timer_callback, this));
    _pose_sub = this->create_subscription<turtlesim::msg::Pose>(
        "/turtle1/pose", 10,
        std::bind(&TurtleSquareNode::pose_sub_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Turtle Square Node Started");
}

void TurtleSquareNode::timer_callback(){
    cmd_vel_pub();
}

void TurtleSquareNode::pose_sub_callback(const turtlesim::msg::Pose::SharedPtr msg){
    _pose_now.x = msg->x;
    _pose_now.y = msg->y;
    _pose_now.theta = msg->theta;
}

void TurtleSquareNode::cmd_vel_pub(){
    const float DISTANCE_TOLERANCE = 0.1;
    const float ANGLE_TOLERANCE = 0.05;
    const float LINEAR_SPEED = 1.0;
    const float ANGULAR_SPEED = 1.0;

    float* goal_point = _through_point[_current_point_index];
    float dx = goal_point[0] - _pose_now.x;
    float dy = goal_point[1] - _pose_now.y;
    float distance = std::sqrt(dx * dx + dy * dy);
    float angle_to_goal = std::atan2(dy, dx);
    float angle_diff = angle_to_goal - _pose_now.theta;
    
    // Normalize angle difference to [-pi, pi]
    while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
    while (angle_diff < -M_PI) angle_diff += 2 * M_PI;

    if (_state == MOVING) {
        if (distance < DISTANCE_TOLERANCE) {
            // Reached the point, move to next point
            _cmd_vel.linear.x = 0.0;
            _cmd_vel.angular.z = 0.0;
            _cmd_vel_pub->publish(_cmd_vel);
            
            _current_point_index = (_current_point_index + 1) % 4;
            _state = TURNING;
            
            // Calculate target angle for next point
            float* next_goal = _through_point[_current_point_index];
            float next_dx = next_goal[0] - _pose_now.x;
            float next_dy = next_goal[1] - _pose_now.y;
            _target_theta = std::atan2(next_dy, next_dx);
            
            RCLCPP_INFO(this->get_logger(), "Reached point %d, turning to next point", (_current_point_index + 3) % 4);
        } else {
            // Still moving towards the point
            if (std::abs(angle_diff) > ANGLE_TOLERANCE) {
                // Need to adjust angle while moving
                _cmd_vel.linear.x = LINEAR_SPEED * 0.5;
                _cmd_vel.angular.z = ANGULAR_SPEED * (angle_diff > 0 ? 1.0 : -1.0);
            } else {
                // Moving straight
                _cmd_vel.linear.x = LINEAR_SPEED;
                _cmd_vel.angular.z = 0.0;
            }
            _cmd_vel_pub->publish(_cmd_vel);
        }
    } else if (_state == TURNING) {
        float turn_angle_diff = _target_theta - _pose_now.theta;
        while (turn_angle_diff > M_PI) turn_angle_diff -= 2 * M_PI;
        while (turn_angle_diff < -M_PI) turn_angle_diff += 2 * M_PI;
        
        if (std::abs(turn_angle_diff) < ANGLE_TOLERANCE) {
            // Finished turning, start moving
            _cmd_vel.linear.x = 0.0;
            _cmd_vel.angular.z = 0.0;
            _cmd_vel_pub->publish(_cmd_vel);
            _state = MOVING;
            RCLCPP_INFO(this->get_logger(), "Moving to point %d", _current_point_index);
        } else {
            // Still turning
            _cmd_vel.linear.x = 0.0;
            _cmd_vel.angular.z = ANGULAR_SPEED * (turn_angle_diff > 0 ? 1.0 : -1.0);
            _cmd_vel_pub->publish(_cmd_vel);
        }
    }
}


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TurtleSquareNode>());
  rclcpp::shutdown();
  return 0;
}