from controller import Supervisor
import os
import numpy as np
import LLM_decider
from heatmap_utils import HeatmapRecorder
from navigation_utils import (
    CORRIDOR_ENTRY_GRACE_STEPS,
    CORRIDOR_EXIT_CONFIRM_STEPS,
    CORRIDOR_HOLD_STEPS,
    DECISION_INTERVAL,
    FRONT_CLEAR,
    FRONT_CRITICAL,
    LOG_INTERVAL,
    MAX_SPEED,
    MAX_STEPS,
    RECOVERY_STEPS,
    SIDE_DANGER,
    SIDE_WARN,
    STUCK_DISTANCE,
    STUCK_RECOVERY_STEPS,
    STUCK_WINDOW,
    TIME_STEP,
    TURN_DEADBAND,
    compute_front_blocked,
    compute_side_too_close,
    compute_state1_controls,
    deadband,
    is_corridor_candidate,
    is_corridor_entry_candidate,
    is_corridor_inside_valid,
    local_state_decision,
    parse_state_answer,
    validate_state_decision,
)
from terminal_utils import (
    print_change,
    print_decision_event,
    print_status,
    print_stuck_event,
)


with open("api_keys.txt", "r") as f:
    content = f.readline()
API_KEY = content.split("=")[1]

robot = Supervisor()
node = robot.getFromDef("PIONEER")

front_left_wheel = robot.getDevice("front left wheel")
front_right_wheel = robot.getDevice("front right wheel")
back_left_wheel = robot.getDevice("back left wheel")
back_right_wheel = robot.getDevice("back right wheel")

for wheel in [front_left_wheel, front_right_wheel, back_left_wheel, back_right_wheel]:
    wheel.setPosition(float("inf"))
    wheel.setVelocity(0)

lidar = robot.getDevice("lidar")
lidar.enable(TIME_STEP)
max_range = lidar.getMaxRange()

step_count = 0
state = 0
state_start_step = 0
front_velocity = np.random.uniform(0.5, 1)
wall_repulsion = 0.1

recovery_steps = 0
recovery_turn_direction = 1
corridor_hold_until = 0
corridor_invalid_steps = 0

heatmap = HeatmapRecorder()
LLM_assistant = LLM_decider.LLMDecider(API_KEY)

while robot.step(TIME_STEP) != -1 and step_count < MAX_STEPS:
    step_count += 1

    scan = np.array(lidar.getRangeImage())
    scan[np.isinf(scan)] = max_range
    left = np.min(scan[85:110])
    left_front = np.min(scan[110:160])
    front = np.min(scan[160:200])
    right_front = np.min(scan[200:250])
    right = np.min(scan[250:275])

    sensors = np.array([left, left_front, front, right_front, right])
    left_side = (left * 0.6 + left_front * 0.4) / max_range
    right_side = (right * 0.6 + right_front * 0.4) / max_range
    free_space = (left_side - right_side) * 2
    front_cover = front / max_range

    front_blocked_now = compute_front_blocked(left_front, front, right_front)
    side_warning_now = compute_side_too_close(left, right)
    corridor_now = is_corridor_candidate(left, left_front, front, right_front, right)
    corridor_entry = is_corridor_entry_candidate(left, left_front, front, right_front, right)
    corridor_inside = is_corridor_inside_valid(left, left_front, front, right_front, right)
    corridor_hint = corridor_entry or (
        min(left, right) > SIDE_DANGER
        and max(left, right) > 0.90
        and min(left_front, right_front) > 0.35
    )

    if corridor_entry and state != 1 and recovery_steps == 0:
        previous_state = state
        state = 1
        state_start_step = step_count
        corridor_hold_until = step_count + CORRIDOR_HOLD_STEPS
        corridor_invalid_steps = 0
        print_change(step_count, previous_state, state, "pasillo entrada detectada")

    corridor_hold_valid = corridor_inside

    if state == 1:
        in_entry_grace = step_count - state_start_step < CORRIDOR_ENTRY_GRACE_STEPS
        if corridor_inside or in_entry_grace:
            corridor_invalid_steps = 0
        else:
            corridor_invalid_steps += 1

        if corridor_invalid_steps >= CORRIDOR_EXIT_CONFIRM_STEPS:
            previous_state = state
            state = 0
            state_start_step = step_count
            corridor_hold_until = 0
            corridor_invalid_steps = 0
            print_change(step_count, previous_state, state, "pasillo perdido confirmado")

    if state == 1 and step_count < corridor_hold_until and corridor_hold_valid:
        side_warning_now = False

    lateral_only_now = front > FRONT_CLEAR and min(left_front, right_front) > 1.10

    if state == 2 and (front_blocked_now or (side_warning_now and not lateral_only_now)):
        previous_state = state
        state = 0
        state_start_step = step_count
        print_change(step_count, previous_state, state, "obstaculo en exploracion")

    forward = 0.0
    turn = 0.0

    if recovery_steps > 0:
        turn = 0.35 * recovery_turn_direction
        vel_left = turn
        vel_right = -turn
        recovery_steps -= 1

    elif state == 0:
        if step_count % 500 == 0:
            front_velocity = np.random.uniform(0.5, 1)

        side_crash_now = min(left, right) < SIDE_DANGER and abs(left - right) > 0.25 and front < FRONT_CLEAR

        if front < FRONT_CRITICAL or side_crash_now:
            if corridor_hint:
                forward = 0.0
                turn = 0.0
                vel_left = 0.0
                vel_right = 0.0
            else:
                left_opening = left + left_front
                right_opening = right + right_front
                recovery_turn_direction = -1 if left_opening > right_opening else 1
                recovery_steps = RECOVERY_STEPS
                turn = 0.35 * recovery_turn_direction
                vel_left = turn
                vel_right = -turn
        elif side_warning_now:
            left_clearance = 0.65 * left + 0.35 * left_front
            right_clearance = 0.65 * right + 0.35 * right_front
            front_priority = np.clip(front / FRONT_CLEAR, 0.0, 1.0)
            front_diagonal_clear = min(left_front, right_front) > 1.10
            lateral_only = front > FRONT_CLEAR and front_diagonal_clear

            max_turn = 0.015 if lateral_only else 0.04 + 0.10 * (1.0 - front_priority)
            turn_error = deadband((right_clearance - left_clearance) / max_range, TURN_DEADBAND)
            turn = np.clip(turn_error, -max_turn, max_turn)
            forward = 0.78 if lateral_only else 0.62 + 0.16 * front_priority
            vel_left = forward + turn
            vel_right = forward - turn
        else:
            front_priority = np.clip(front / FRONT_CLEAR, 0.0, 1.0)
            max_turn = 0.08 + 0.17 * (1.0 - front_priority)
            turn_error = deadband(free_space, TURN_DEADBAND)
            turn = np.clip(turn_error * ((1 - front_cover) + wall_repulsion), -max_turn, max_turn)
            vel_left = front_velocity * front_cover - turn
            vel_right = front_velocity * front_cover + turn

    elif state == 1:
        forward, turn = compute_state1_controls(left, left_front, front, right_front, right, max_range)
        vel_left = forward + turn
        vel_right = forward - turn

    elif state == 2:
        obstacle_pressure = max(0.0, 1.0 - front_cover)
        turn_error = deadband(-free_space, TURN_DEADBAND)
        turn = np.clip(turn_error * (obstacle_pressure + 0.18), -0.20, 0.20)
        forward = np.clip(0.75 * front_cover + 0.20, 0.25, 0.95)
        vel_left = forward + turn
        vel_right = forward - turn

    vel_left = np.clip(vel_left, -1, 1) * MAX_SPEED
    vel_right = np.clip(vel_right, -1, 1) * MAX_SPEED

    front_left_wheel.setVelocity(vel_left)
    back_left_wheel.setVelocity(vel_left)
    front_right_wheel.setVelocity(vel_right)
    back_right_wheel.setVelocity(vel_right)

    heatmap.record_node_position(node)

    can_check_stuck = step_count % 40 == 0 and recovery_steps == 0
    if can_check_stuck:
        is_stuck, displacement = heatmap.check_stuck(step_count, STUCK_WINDOW, STUCK_DISTANCE)
        if is_stuck:
            previous_state = state
            state = 0
            state_start_step = step_count

            if front > FRONT_CLEAR:
                recovery_steps = 0
                front_velocity = 0.85
            else:
                recovery_turn_direction = -1 if (left + left_front) > (right + right_front) else 1
                recovery_steps = STUCK_RECOVERY_STEPS

            print_change(step_count, previous_state, state, "atasco")
            print_stuck_event(step_count, displacement, recovery_steps, front)

    front_blocked = compute_front_blocked(left_front, front, right_front)
    side_too_close = compute_side_too_close(left, right)
    decision_moment = step_count % DECISION_INTERVAL == 0

    if state == 1 and front <= FRONT_CRITICAL:
        previous_state = state
        state = 0
        state_start_step = step_count
        corridor_hold_until = 0
        corridor_invalid_steps = 0
        print_change(step_count, previous_state, state, "frente critico en pasillo")
    elif front_blocked and state != 0 and not (
        state == 1
        and step_count < corridor_hold_until
        and (corridor_hold_valid or front > FRONT_CRITICAL)
    ):
        previous_state = state
        state = 0
        state_start_step = step_count
        print_change(step_count, previous_state, state, "frente bloqueado")
    elif side_too_close and state == 2 and not (
        front > FRONT_CLEAR and min(left_front, right_front) > 1.10
    ):
        previous_state = state
        state = 0
        state_start_step = step_count
        print_change(step_count, previous_state, state, "lateral demasiado cerca")

    if decision_moment and step_count - state_start_step >= 25:
        robot_state = {
            "state": state,
            "steps_in_state": step_count - state_start_step,
        }
        answer = LLM_assistant.decide(sensors, robot_state)
        selected_state = parse_state_answer(answer)
        source = "groq"
        groq_answer = "None" if answer is None else answer.strip().upper()

        if selected_state is None:
            selected_state = local_state_decision(left, left_front, front, right_front, right)
            source = "local"
        else:
            selected_state = validate_state_decision(selected_state, left, left_front, front, right_front, right)

        if (
            state == 1
            and step_count < corridor_hold_until
            and (
                corridor_hold_valid
                or step_count - state_start_step < CORRIDOR_ENTRY_GRACE_STEPS
            )
        ):
            selected_state = 1

        if selected_state != state:
            previous_state = state
            state = selected_state
            state_start_step = step_count
            if state == 1:
                corridor_hold_until = step_count + CORRIDOR_HOLD_STEPS
                corridor_invalid_steps = 0
            print_change(step_count, previous_state, state, "decision {}".format(source))

        print_decision_event(step_count, groq_answer, selected_state, state, source)

    if step_count % LOG_INTERVAL == 0:
        print_status(
            step_count,
            state,
            state_start_step,
            sensors,
            corridor_now,
            corridor_entry,
            corridor_inside,
            corridor_invalid_steps,
            front_blocked,
            side_too_close,
            vel_left,
            vel_right,
            forward,
            turn,
            corridor_hold_until,
        )

front_left_wheel.setVelocity(0)
back_left_wheel.setVelocity(0)
front_right_wheel.setVelocity(0)
back_right_wheel.setVelocity(0)

heatmap_path = os.path.abspath("heatmap_recorrido_13.png")
heatmap.finish(heatmap_path)
