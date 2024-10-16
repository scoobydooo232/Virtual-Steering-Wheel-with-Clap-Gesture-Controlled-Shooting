# Virtual-Steering-Wheel-with-Clap-Gesture-Controlled-Shooting

## Overview
This project implements a **virtual steering wheel** for controlling a tank simulation using **hand gestures and head tracking**. Additionally, a **clap gesture** is integrated to shoot bullets from the tank. The system is developed using **OpenCV** and **MediaPipe**, which allow real-time detection of hand, pose, and facial landmarks, making it possible to control the tank seamlessly through physical gestures.

## Features
- **Hand Gesture Steering**: 
    - Control the steering of the virtual tank using hand positions. The system detects the relative position of both hands and calculates the angle to determine turning actions.
    - Turning actions are mapped to real-time keypresses for steering (left and right turns).
- **Head Tracking for Cursor Movement**:
    - The head position is tracked using the MediaPipe Face Mesh, and the nose's position relative to the eyes is used to simulate cursor movement or further control the vehicle.
- **Clap Gesture for Shooting**:
    - When the system detects the hands clapping, it triggers the shooting of bullets from the tank by simulating a keypress.
- **Acceleration and Braking**:
    - Detects whether the right or left hand is closed to control acceleration or braking.
    
## How It Works
1. **Hand Gesture Recognition**: 
    - The system uses **MediaPipe Hands** to recognize the position of both hands and calculates the distance and angle between them. This angle is used to determine the steering direction.
    - A hard left or right turn is performed when the angle crosses a certain threshold.
2. **Head Tracking**: 
    - **MediaPipe Face Mesh** is used to track facial landmarks and detect the head's rotation. The horizontal movement of the nose relative to the eyes is translated into cursor movements, enhancing the control experience.
3. **Clap Gesture Detection**:
    - When the system detects a clap (when both hands disappear briefly and then reappear), it simulates a bullet shooting action by pressing a designated key ('i' key in this case).
4. **Direct Input for Vehicle Control**:
    - **ctypes** is used to simulate keyboard inputs for controlling the tank. Keys for forward, backward, left, right, and shooting are mapped to detected gestures.

## Dependencies
- OpenCV
- MediaPipe
- Python `ctypes`
- NumPy
- PyAutoGUI (for additional input handling)
