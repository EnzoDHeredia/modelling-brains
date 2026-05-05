import os
from groq import Groq


class LLMDecider:

    def __init__(self, api_key, model="llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def decide(self, sensor_state):
        if self.client is None:
            return None

        if len(sensor_state) < 5:
            return None

        left, left_front, front, right_front, right = [float(v) for v in sensor_state[:5]]
        left_score = 0.40 * left + 0.60 * left_front
        right_score = 0.40 * right + 0.60 * right_front

        system_prompt = (
            "You are a robot navigation supervisor.\n"
            "Return exactly one token: LEFT or RIGHT.\n"
            "Sensor order is: Left, FrontLeft, Front, FrontRight, Right.\n"
            "Sensor values range from 0.0 to 2.0:\n"
            "- 0.0 means wall is very close\n"
            "- 2.0 means free space\n"
            "Choose the safer turn direction (away from closer walls), preferring the side with more free space."
        )

        user_prompt = (
            "Choose LEFT or RIGHT.\n"
            f"Sensors: Left={left:.2f}; FrontLeft={left_front:.2f}; Front={front:.2f}; "
            f"FrontRight={right_front:.2f}; Right={right:.2f}\n"
            f"Computed free-space scores: left_score={left_score:.3f}, right_score={right_score:.3f}\n"
            "Output one token only."
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2,
            )
        except Exception:
            return None

        raw = completion.choices[0].message.content if completion and completion.choices else ""
        self.last_raw_answer = raw
        parsed = (raw or "").strip().upper()
        token = parsed.split()[0] if parsed else ""


        return token


