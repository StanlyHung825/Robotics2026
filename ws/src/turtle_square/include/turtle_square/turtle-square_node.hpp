#ifndef TURTLE_SQUARE_NODE_HPP
#define TURTLE_SQUARE_NODE_HPP

#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "turtlesim/msg/pose.hpp"

class TurtleSquareNode : public rclcpp::Node
{
    public:
        TurtleSquareNode();
    
    private:
        void timer_callback();
        void pose_sub_callback(const turtlesim::msg::Pose::SharedPtr msg);
        void cmd_vel_pub();
        
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr _cmd_vel_pub;
        rclcpp::TimerBase::SharedPtr _timer;
        rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr _pose_sub;
        geometry_msgs::msg::Twist _cmd_vel;
        float _through_point[4][2] = {{2.0, 2.0}, {8.0, 2.0}, {8.0, 8.0}, {2.0, 8.0}};
        int _current_point_index = 0;
        turtlesim::msg::Pose _pose_now;
        enum State { MOVING, TURNING };
        State _state = MOVING;
        float _target_theta = 0.0;
};

#endif