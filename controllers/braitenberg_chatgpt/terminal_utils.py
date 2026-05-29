from navigation_utils import STATE_NAMES


def format_bool(value):
    return "yes" if value else "no"


def print_log_block(title, rows):
    width = 72
    print("=" * width)
    print(title)
    print("-" * width)
    for label, value in rows:
        print("{:<9} {}".format(label, value))
    print("=" * width)


def print_state_change(step, old_state, new_state, reason):
    if old_state == new_state:
        return

    print_log_block(
        "EVENT step={:04d} | STATE CHANGE".format(step),
        [
            ("FROM", "{} {}".format(old_state, STATE_NAMES[old_state])),
            ("TO", "{} {}".format(new_state, STATE_NAMES[new_state])),
            ("REASON", reason),
        ],
    )


def print_decision(step, groq_answer, selected_state, applied_state, source):
    print_log_block(
        "EVENT step={:04d} | DECISION".format(step),
        [
            ("GROQ", groq_answer),
            ("SELECTED", selected_state),
            ("APPLIED", applied_state),
            ("SOURCE", source),
        ],
    )


def print_stuck_event(step, displacement, recovery_steps, front):
    print_log_block(
        "EVENT step={:04d} | STUCK".format(step),
        [
            ("MOVE", "desplazamiento={:.3f}".format(displacement)),
            ("RECOVERY", "steps={}".format(recovery_steps)),
            ("FRONT", "{:.2f}".format(front)),
        ],
    )


def print_robot_status(
    step,
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
):
    left, left_front, front, right_front, right = sensors
    hold_active = state == 1 and step < corridor_hold_until
    lateral_error = right - left

    rows = [
        (
            "SENSORS",
            "L={:.2f}  LF={:.2f}  F={:.2f}  RF={:.2f}  R={:.2f}".format(
                left,
                left_front,
                front,
                right_front,
                right,
            ),
        ),
        (
            "FLAGS",
            "entry={}  inside={}  strict={}  hold={}  front_bad={}  side_close={}".format(
                format_bool(corridor_entry),
                format_bool(corridor_inside),
                format_bool(corridor_now),
                format_bool(hold_active),
                format_bool(front_blocked),
                format_bool(side_too_close),
            ),
        ),
        (
            "CONTROL",
            "left={:.2f}  right={:.2f}  forward={:.2f}  turn={:.2f}".format(
                vel_left,
                vel_right,
                forward,
                turn,
            ),
        ),
    ]

    if state == 1:
        rows.append(
            (
                "CORRIDOR",
                "invalid={}  hold_until={}  lateral_error={:.2f}".format(
                    corridor_invalid_steps,
                    corridor_hold_until,
                    lateral_error,
                ),
            )
        )

    print_log_block(
        "STEP {:04d} | STATE {} {} | in_state={:03d}".format(
            step,
            state,
            STATE_NAMES[state],
            step - state_start_step,
        ),
        rows,
    )


print_change = print_state_change
print_decision_event = print_decision
print_status = print_robot_status
