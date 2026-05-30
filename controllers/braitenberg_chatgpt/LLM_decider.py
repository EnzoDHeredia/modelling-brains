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
        open_space = min(left, left_front, front, right_front, right) > 1.35
        corridor_candidate = (
            front >= 0.40
            and left_front > 0.30
            and right_front > 0.30
            and 0.30 < left < 1.60
            and 0.30 < right < 1.60
            and abs(left - right) < 0.95
            and abs(left_front - right_front) < 1.10
        )
        critical_obstacle = (
            front < 0.40
            or left <= 0.30
            or right <= 0.30
            or left_front <= 0.30
            or right_front <= 0.30
        )
    
        current_state = 0
        steps_in_state = 0
        is_stuck = False

        if robot_state:
            current_state  = int(robot_state.get("state", 0))
            steps_in_state = int(robot_state.get("steps_in_state", 0))
            is_stuck       = bool(robot_state.get("is_stuck", False))
    
        system_prompt = (
            "You are a robot navigation supervisor.\n"
            "Return exactly one token: STATE_0, STATE_1, or STATE_2.\n"
            "Sensor order: Left, FrontLeft, Front, FrontRight, Right. Range 0.0 to 2.0 (0=wall, 2=free space).\n\n"
            "Available behaviours:\n"
            "STATE_0 = Braitenberg obstacle avoidance for blocked or risky situations.\n"
            "STATE_1 = Narrow corridor navigation, centered between two lateral walls.\n"
            "STATE_2 = Exploration in fully open space.\n\n"
            "Use the derived flags as authoritative. Do not reinterpret close side walls as danger when corridor_candidate=True.\n"
            "Priority rules:\n"
            "1. If is_stuck=True or critical_obstacle=True, return STATE_0.\n"
            "2. Else if open_space=True, return STATE_2.\n"
            "3. Else if corridor_candidate=True, return STATE_1.\n"
            "4. Else return STATE_0.\n"
            "Examples:\n"
            "- Sensors Left=0.35, FrontLeft=0.35, Front=0.64, FrontRight=0.81, Right=0.67 => STATE_1.\n"
            "- Sensors Left=0.75, FrontLeft=1.08, Front=0.44, FrontRight=0.34, Right=0.35 => STATE_1.\n"
            "Do not explain your choice."
        )
    
        user_prompt = (
            f"Current state={current_state}; steps_in_state={steps_in_state}; is_stuck={is_stuck}\n"
            f"Sensors: Left={left:.2f}; FrontLeft={left_front:.2f}; Front={front:.2f}; "
            f"FrontRight={right_front:.2f}; Right={right:.2f}\n"
            f"Derived flags: open_space={open_space}; corridor_candidate={corridor_candidate}; "
            f"critical_obstacle={critical_obstacle}\n"
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
    
    
    
