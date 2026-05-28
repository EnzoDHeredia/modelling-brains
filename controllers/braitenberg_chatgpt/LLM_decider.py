import os
from groq import Groq


class LLMDecider:

    def __init__(self, api_key, model="llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def decide(self, sensor_state, robot_state=None):
        if self.client is None:
            return None
        if len(sensor_state) < 5:
            return None
    
        left, left_front, front, right_front, right = [float(v) for v in sensor_state[:5]]
        left_score = 0.40 * left + 0.60 * left_front
        right_score = 0.40 * right + 0.60 * right_front
        lateral_clearance = min(left, right)
        front_clearance = min(left_front, front, right_front)
        corridor_score = min(left, right) - abs(left - right)
        open_space_score = min(sensor_state)
    
        current_state = 0
        steps_in_state = 0
        is_stuck = False
        vel_left = 0.0
        vel_right = 0.0
    
        if robot_state:
            current_state  = int(robot_state.get("state", 0))
            steps_in_state = int(robot_state.get("steps_in_state", 0))
            is_stuck       = bool(robot_state.get("is_stuck", False))
            vel_left       = float(robot_state.get("vel_left", 0.0))
            vel_right      = float(robot_state.get("vel_right", 0.0))
    
        system_prompt = (
            "You are a robot navigation supervisor.\n"
            "Return exactly one token: STATE_0, STATE_1, or STATE_2.\n"
            "Sensor order: Left, FrontLeft, Front, FrontRight, Right. Range 0.0 to 2.0 (0=wall, 2=free space).\n\n"
            "Available behaviours:\n"
            "STATE_0 = Braitenberg obstacle avoidance. "
            "Reactive behaviour: steers away from nearby obstacles using sensor difference. "
            "Use when front or sides are dangerously close.\n"
            "STATE_1 = Narrow corridor navigation. "
            "Keeps the robot centered between two lateral walls while moving forward steadily. "
            "Use when both sides are detected and the front is still open.\n"
            "STATE_2 = Systematic exploration sweep. "
            "Robot alternates turn direction every fixed number of steps to cover new map areas. "
            "Use only when all sensors read above 1.35 and the space is fully open.\n\n"
            "Decision rules:\n"
            "- STATE_1: Front > 0.95, 0.40 < Left < 1.60, 0.40 < Right < 1.60, abs(Left-Right) < 0.95.\n"
            "- STATE_2: all sensors > 1.35. Robot is in fully open space with no nearby walls.\n"
            "- STATE_0: any other situation, especially when front is blocked or any side is critical.\n"
            "- Never choose STATE_2 if any sensor is below 1.35.\n"
            "- Never choose STATE_1 if Front < 0.45 or either side is below 0.40.\n"
            "- If the robot is stuck, always prefer STATE_0.\n"
            "Do not explain your choice."
        )
    
        user_prompt = (
            f"Current state={current_state}; steps_in_state={steps_in_state}; is_stuck={is_stuck}\n"
            f"Wheel velocities (normalized -1 to 1): left={vel_left:.2f}, right={vel_right:.2f}\n"
            f"Sensors: Left={left:.2f}; FrontLeft={left_front:.2f}; Front={front:.2f}; "
            f"FrontRight={right_front:.2f}; Right={right:.2f}\n"
            f"Computed scores: left_score={left_score:.3f}, right_score={right_score:.3f}, "
            f"front_clearance={front_clearance:.3f}, lateral_clearance={lateral_clearance:.3f}, "
            f"corridor_score={corridor_score:.3f}, open_space_score={open_space_score:.3f}\n"
            "Output one token only: STATE_0, STATE_1, or STATE_2."
        )
    
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=5,
            )
        except Exception:
            return None
    
        raw = completion.choices[0].message.content if completion and completion.choices else ""
        self.last_raw_answer = raw
        parsed = (raw or "").strip().upper()
        token = parsed.split()[0] if parsed else ""
        return token
    
    
    