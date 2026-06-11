# 11420PME-526000-Robotics-LAB2

## LAB2

- Due: 2026/03/30

## Prerequisites
- Go to the GitHub Marketplace and generate your own token: [GitHub Models](https://github.com/marketplace/models/azure-openai/gpt-5)
- Example video: https://github.com/user-attachments/assets/615c8edb-f759-42a4-965b-21c3c3218915

## Workspace Setup

1. Download `voicegpt.zip` from eLearn.
2. Unzip it into `<workspace>/src`.
3. Build the workspace:

```bash
colcon build --symlink-install
```

Make sure the build output is generated under your workspace.

4. Export your GitHub token:

```bash
export GITHUB_TOKEN="YOUR-GITHUB-TOKEN-GOES-HERE"
```

5. Install dependencies under `<workspace>/src/voicegpt`:

```bash
cd <workspace>/src/voicegpt
bash install.sh
```

6. Verify that the package exists:

```bash
ros2 pkg prefix voicegpt
```

## Run

1. Source your workspace environment:

```bash
source <workspace>/install/setup.bash
```

2. Launch the full demo:

```bash
ros2 launch voicegpt lab2.launch.py
```

3. Speak a command after the microphone starts listening.

Before modifying the code, this starter version supports only simple `move` commands. It does not yet support more complex behaviors such as drawing shapes, multi-step planning, or richer motion types.

Example voice commands:

- `Move forward 1 meter with speed 0.5 meters per second.`
- `Move backward 0.5 meters with speed 0.2 meters per second.`

The system recognizes your speech, sends it to GPT, publishes the JSON command to `/gpt_reply_to_user`, and then drives `turtlesim` through `/turtle1/cmd_vel`.

## Architecture

### Components

| Component | Role | Input | Output |
| --- | --- | --- | --- |
| `voicegpt.py` | Captures speech, performs speech recognition, sends the prompt to GPT, and publishes the GPT result to ROS 2. | Microphone audio | `/gpt_reply_to_user` (`std_msgs/String`) |
| `turtlenode.py` | Subscribes to GPT JSON commands, parses motion parameters, and converts them into velocity commands. | `/gpt_reply_to_user` (`std_msgs/String`) | `/turtle1/cmd_vel` (`geometry_msgs/Twist`) |
| `turtlesim` | Simulates the turtle and executes received velocity commands. | `/turtle1/cmd_vel` (`geometry_msgs/Twist`) | Turtle motion in simulation |

### Internal Processing

1. `voicegpt.py` records speech from the microphone.
2. Google Speech Recognition converts the audio into text.
3. The recognized text is sent to GitHub Models (`openai/gpt-4.1`).
4. GPT returns a JSON command such as `{"action":"move","linear_x":0.5,"distance":1.0}`.
5. `voicegpt.py` publishes the JSON string to `/gpt_reply_to_user`.
6. `turtlenode.py` parses the JSON command and publishes a `Twist` message to `/turtle1/cmd_vel`.

## Tips

- You may encounter **free-tier rate limits** during testing, so it is recommended that **each team member create their own token** for smoother testing.
- When using **voice input**, a **headset microphone** is recommended for better recognition performance. After you finish speaking, **mute or turn off the microphone** to help ensure the **full prompt is captured correctly**.

## Lab Requirements

Your `voicegpt` code should be extended to control the turtle along a square path with random speed and length.
