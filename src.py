import cv2
import mediapipe as mp
import numpy as np
import math
import ctypes
import time
import pyautogui

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Direct input setup
SendInput = ctypes.windll.user32.SendInput
A = 0x1E
D = 0x20
W = 0x11
S = 0x1F
KEY_I = 0x49  # Virtual Key Code for '9'

# C struct redefinitions for direct input
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actual functions for direct input
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# Function to send raw mouse movement input
def move_mouse(dx, dy):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(dx, dy, 0, 0x0001, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# Function to check if a hand is closed
def is_hand_closed(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    closed = (
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )

    return closed

# Function to perform turning action with adjustable key hold time
def perform_turn(key, hold_time):
    PressKey(key)  # Press the key
    time.sleep(hold_time)  # Hold the key for a specified duration
    ReleaseKey(key)  # Release the key

# Function to simulate pressing the 'i' key
def press_i_key():
    PressKey(KEY_I)
    time.sleep(0.1)  # Short delay
    ReleaseKey(KEY_I)

# Timer to track the duration for which hands are not detected
hand_not_detected_start_time = None
clap_delay = 2  # 2 seconds delay

# Function to detect head tud9d9ddd99aa9aaaaarn and move cursor
def get_head_turn(landmarks):
    # Head turn (left-right)
    nose = landmarks[1]         # Nose tip
    left_eye = landmarks[33]    # Left eye
    right_eye = landmarks[263]  # Right eye

    # Midpoint between the eyes
    midpoint_x = (left_eye.x + right_eye.x) / 2

    # Calculate the horizontal difference between the nose and the midpoint
    nose_diff_x = nose.x - midpoint_x

    return nose_diff_x

def move_cursor_based_on_turn(nose_diff_x):
    sensitivity = 80  # Adjust sensitivity for raw movement
    threshold = 0.02  # Threshold for detecting a significant turn

    # Inverted logic for cursor movement direction
    if nose_diff_x > threshold:  # Turned right, move mouse left
        move_mouse(sensitivity, 0)
    elif nose_diff_x < -threshold:  # Turned left, move mouse right
        move_mouse(-sensitivity, 0)

# Open webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for pose landmarks
    pose_results = pose.process(frame_rgb)
    
    # Process the frame for hand landmarks
    hands_results = hands.process(frame_rgb)
    
    # Process the frame for face landmarks (for head tracking)
    face_results = face_mesh.process(frame_rgb)
    
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    hand_landmarks_list = []
    
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_coords = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
            hand_landmarks_list.append((wrist_coords, hand_landmarks))
    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            
            nose_diff_x = get_head_turn(face_landmarks.landmark)
            move_cursor_based_on_turn(nose_diff_x)
    
    if len(hand_landmarks_list) == 2:
        # Determine the right hand based on the x-coordinate
        frame_width = frame.shape[1]
        middle_x = frame_width // 2
        
        # Hand coordinates and landmarks
        (hand1, landmarks1), (hand2, landmarks2) = hand_landmarks_list
        
        if hand1[0] < middle_x and hand2[0] > middle_x:
            right_hand = hand2
            left_hand = hand1
            right_landmarks = landmarks2
            left_landmarks = landmarks1
        elif hand2[0] < middle_x and hand1[0] > middle_x:
            right_hand = hand1
            left_hand = hand2
            right_landmarks = landmarks1
            left_landmarks = landmarks2
        else:
            right_hand = hand1 if hand1[0] > hand2[0] else hand2
            left_hand = hand2 if hand1[0] > hand2[0] else hand1
            right_landmarks = landmarks1 if hand1[0] > hand2[0] else landmarks2
            left_landmarks = landmarks2 if hand1[0] > hand2[0] else landmarks1
        
        # Draw a line between wrists
        cv2.line(frame, left_hand, right_hand, (0, 255, 0), 3)
        
        x1, y1 = left_hand
        x2, y2 = right_hand
        
        # Calculate distance
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Calculate the angle with respect to positive x-axis
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Calculate midpoint
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        # Draw horizontal line from the midpoint
        cv2.line(frame, (mid_x, mid_y), (frame_width, mid_y), (255, 0, 0), 2)
        
        # Display the distance and angle on the frame
        cv2.putText(frame, f'Distance: {distance:.2f} px', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Angle: {angle:.2f} degrees', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Determine if accelerating or braking
        accelerating = is_hand_closed(right_landmarks)
        braking = is_hand_closed(left_landmarks)

        # Control turning first
        if angle > 30:
            perform_turn(D, 0.1)  # Hard right
            print("Hard Right")
        elif 10 < angle <= 30:
            perform_turn(D, 0.05)  # Slight right
            print("Slight Right")
        elif -30 <= angle < -10:
            perform_turn(A, 0.05)  # Slight left
            print("Slight Left")
        elif angle < -30:
            perform_turn(A, 0.1)  # Hard left
            print("Hard Left")

        # Control acceleration and braking if not turning
        if not (angle > 10 or angle < -10):
            if accelerating:
                PressKey(W)
                print("Accelerate")
            else:
                ReleaseKey(W)
            if braking:
                PressKey(S)
                print("Brake")
            else:
                ReleaseKey(S)
    
    # Check if hands are detected
    if not hands_results.multi_hand_landmarks or len(hands_results.multi_hand_landmarks) < 2:
        if hand_not_detected_start_time is None:
            # Start the timer if it's the first time hands are not detected
            hand_not_detected_start_time = time.time()
    else:
        if hand_not_detected_start_time is not None:
            # Check if hands have been missing for less than 2 seconds
            elapsed_time = time.time() - hand_not_detected_start_time
            if elapsed_time < clap_delay:
                # Simulate pressing the 'i' key
                press_i_key()
                print("Clap detected, 'i' key pressed")
            # Reset the timer if hands are detected
            hand_not_detected_start_time = None

    # Show the frame
    cv2.imshow('Pose and Gesture Controlled Steering with Head Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
