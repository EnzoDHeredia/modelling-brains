import numpy as np


TIME_STEP = 64
MAX_SPEED = 6.0
MAX_STEPS = 10000
DECISION_INTERVAL = 40
LOG_INTERVAL = 40
RECOVERY_STEPS = 5
STUCK_RECOVERY_STEPS = 14
STUCK_WINDOW = 80
STUCK_DISTANCE = 0.12
FRONT_DANGER = 0.90
FRONT_CRITICAL = 0.45
FRONT_CLEAR = 1.35
TURN_DEADBAND = 0.18
DIAGONAL_DANGER = 0.70
SIDE_WARN = 0.85
SIDE_DANGER = 0.35
SIDE_CRITICAL = 0.40
CORRIDOR_MIN_SIDE = 0.40
CORRIDOR_MAX_SIDE = 1.35
CORRIDOR_HOLD_STEPS = 180
CORRIDOR_ENTRY_FRONT_MIN = 1.05
CORRIDOR_STRAIGHT_DEADBAND = 0.30
CORRIDOR_MAX_SOFT_TURN = 0.035
CORRIDOR_ENTRY_SPEED = 0.62
CORRIDOR_ENTRY_GRACE_STEPS = 35
CORRIDOR_EXIT_CONFIRM_STEPS = 5
CORRIDOR_FRONT_STOP = 0.55
CORRIDOR_FORCE_FRONT_MIN = 0.45
CORRIDOR_HOLD_DIAGONAL_MIN = 0.40

STATE_NAMES = {
    0: "Braitenberg/evasion",
    1: "Pasillo angosto",
    2: "Exploracion",
}


def deadband(value, threshold):
    if abs(value) < threshold:
        return 0.0

    return value


def parse_state_answer(answer):
    if answer is None:
        return None

    answer = answer.strip().upper()
    if "STATE_0" in answer or answer == "0":
        return 0
    if "STATE_1" in answer or answer == "1":
        return 1
    if "STATE_2" in answer or answer == "2":
        return 2

    return None


def is_corridor_candidate(left, left_front, front, right_front, right):
    if front < CORRIDOR_FORCE_FRONT_MIN:
        return False

    straight_entry = (
        front > CORRIDOR_ENTRY_FRONT_MIN
        and min(left_front, right_front) > 0.60
        and CORRIDOR_MIN_SIDE < min(left, right)
        and max(left, right) < 1.85
        and (
            min(left, right) < 1.25
            or abs(left - right) < 0.75
        )
    )
    side_pair = (
        CORRIDOR_MIN_SIDE < left < 1.60
        and CORRIDOR_MIN_SIDE < right < 1.60
        and abs(left - right) < 0.95
    )
    diagonal_pair = (
        left_front > 0.45
        and right_front > 0.45
        and abs(left_front - right_front) < 1.10
    )
    funnel_entry = (
        front > 1.20
        and min(left, right) < 0.80
        and max(left, right) < 1.65
        and min(left_front, right_front) > 0.45
    )
    tight_entry = (
        front > CORRIDOR_FORCE_FRONT_MIN
        and min(left, right) > SIDE_CRITICAL
        and max(left, right) > 1.00
        and max(left, right) < 1.70
        and min(left_front, right_front) > 0.55
    )
    bottom_corridor_entry = (
        front > CORRIDOR_FORCE_FRONT_MIN
        and min(left, right) > SIDE_CRITICAL
        and max(left, right) > 0.90
        and max(left, right) < 1.70
        and min(left_front, right_front) > 0.45
    )

    return straight_entry or (side_pair and diagonal_pair) or funnel_entry or tight_entry or bottom_corridor_entry


def is_corridor_entry_candidate(left, left_front, front, right_front, right):
    if front <= FRONT_CRITICAL:
        return False

    diagonal_clear = min(left_front, right_front) > 0.35
    bounded_sides = min(left, right) > SIDE_DANGER and max(left, right) < 2.10
    asymmetric_entry = (
        front > CORRIDOR_FRONT_STOP
        and diagonal_clear
        and bounded_sides
        and min(left, right) < SIDE_WARN
        and max(left, right) > 0.90
    )
    straight_entry = (
        front > CORRIDOR_ENTRY_FRONT_MIN
        and diagonal_clear
        and bounded_sides
    )

    return is_corridor_candidate(left, left_front, front, right_front, right) or asymmetric_entry or straight_entry


def is_corridor_inside_valid(left, left_front, front, right_front, right):
    return (
        front > FRONT_CRITICAL
        and min(left, right) >= SIDE_DANGER
        and min(left_front, right_front) >= 0.35
        and max(left, right) < 2.15
    )


def local_state_decision(left, left_front, front, right_front, right):
    mostly_open = min(left, left_front, front, right_front, right) > 1.35

    if is_corridor_entry_candidate(left, left_front, front, right_front, right):
        return 1
    if mostly_open:
        return 2
    return 0


def validate_state_decision(selected_state, left, left_front, front, right_front, right):
    front_blocked = compute_front_blocked(left_front, front, right_front)
    side_too_close = compute_side_too_close(left, right)

    if selected_state == 1 and not is_corridor_entry_candidate(left, left_front, front, right_front, right):
        return 0
    if selected_state == 2 and (front_blocked or side_too_close):
        return 0

    return selected_state


def compute_front_blocked(left_front, front, right_front):
    return front < FRONT_DANGER or left_front < DIAGONAL_DANGER or right_front < DIAGONAL_DANGER


def compute_side_too_close(left, right):
    return left < SIDE_WARN or right < SIDE_WARN


def compute_corridor_hold_valid(left, left_front, front, right_front, right):
    return is_corridor_inside_valid(left, left_front, front, right_front, right)


def compute_state1_controls(left, left_front, front, right_front, right, max_range):
    if front <= FRONT_CRITICAL:
        return 0.0, 0.0

    side_error = (right - left) / max_range
    turn = 0.0

    if front > FRONT_CLEAR and min(left_front, right_front) > 1.00:
        forward = CORRIDOR_ENTRY_SPEED
    elif front > CORRIDOR_ENTRY_FRONT_MIN:
        forward = 0.52
    else:
        forward = 0.34

    if abs(side_error) >= CORRIDOR_STRAIGHT_DEADBAND or min(left, right) < SIDE_WARN:
        turn = np.clip(side_error * 0.18, -CORRIDOR_MAX_SOFT_TURN, CORRIDOR_MAX_SOFT_TURN)

    if front < CORRIDOR_FRONT_STOP:
        turn = 0.0
        forward = 0.08
    elif front < 0.90:
        turn += np.clip((right_front - left_front) / max_range, -CORRIDOR_MAX_SOFT_TURN, CORRIDOR_MAX_SOFT_TURN)
        forward = min(forward, 0.30)

    return forward, np.clip(turn, -CORRIDOR_MAX_SOFT_TURN, CORRIDOR_MAX_SOFT_TURN)
