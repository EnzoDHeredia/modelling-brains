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
        if robot_state:
            current_state = int(robot_state.get("state", 0))
            steps_in_state = int(robot_state.get("steps_in_state", 0))

        system_prompt = (
            "You are a robot navigation supervisor.\n"
            "Return exactly one token: STATE_0, STATE_1, or STATE_2.\n"
            "Sensor order is: Left, FrontLeft, Front, FrontRight, Right.\n"
            "Sensor values range from 0.0 to 2.0:\n"
            "- 0.0 means wall is very close\n"
            "- 2.0 means free space\n"
            "Available robot behaviours:\n"
            "STATE_0 = Braitenberg obstacle avoidance. Use when an obstacle is close in front or on one side.\n"
            "STATE_1 = narrow corridor crossing. Use when both side walls are detected and the front is still open; "
            "this behaviour keeps the robot centered and moving forward through tight passages.\n"
            "STATE_2 = exploration sweep. Use when the robot sees mostly free space or has not detected walls recently; "
            "this behaviour makes the robot search new parts of the map instead of only avoiding objects.\n"
            "Decision policy:\n"
            "- Choose STATE_1 when the robot is inside or entering a corridor: Front > 0.95, "
            "0.40 < Left < 1.60, 0.40 < Right < 1.60, and abs(Left - Right) < 0.95.\n"
            "- Also choose STATE_1 for a corridor entrance if Front > 1.20, one side is below 0.80, "
            "the other side is below 1.65, and both front diagonals are above 0.45.\n"
            "- For very narrow entrances, choose STATE_1 if Front > 0.45, both sides are above 0.40, "
            "both front diagonals are above 0.45, and the far side is below 1.70.\n"
            "- Never choose STATE_1 if Front < 0.45, either side is below 0.40, either front diagonal is below 0.40, "
            "the far side is very open, or the robot is in a corner.\n"
            "- Never choose STATE_2 if any sensor is below 0.95; use STATE_0 to recover first.\n"
            "- Prefer STATE_1 for narrow corridors, because pure Braitenberg can reject both walls and fail to pass.\n"
            "- Prefer STATE_2 when all directions are open, to increase heat-map coverage.\n"
            "- Prefer STATE_0 when the front is blocked or the situation is asymmetric and collision avoidance is needed.\n"
            "Do not explain your choice."
        )

        user_prompt = (
            "Choose the next behaviour state.\n"
            f"Current state={current_state}; steps_in_state={steps_in_state}\n"
            f"Sensors: Left={left:.2f}; FrontLeft={left_front:.2f}; Front={front:.2f}; "
            f"FrontRight={right_front:.2f}; Right={right:.2f}\n"
            f"Computed free-space scores: left_score={left_score:.3f}, right_score={right_score:.3f}\n"
            f"front_clearance={front_clearance:.3f}; lateral_clearance={lateral_clearance:.3f}; "
            f"corridor_score={corridor_score:.3f}; open_space_score={open_space_score:.3f}\n"
            "Output one token only: STATE_0, STATE_1, or STATE_2."
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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
