from controller import Supervisor
import os
import numpy as np
import LLM_decider
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

##Parametros de simulacion
TIME_STEP = 64
MAX_SPEED = 6.0
MAX_STEPS = 10000
DECISION_INTERVAL = 40
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
CORRIDOR_MAX_IMBALANCE = 0.70
CORRIDOR_MAX_DIAGONAL_IMBALANCE = 0.80
CORRIDOR_FRONT_OPEN = 0.95
CORRIDOR_HOLD_STEPS = 180
CORRIDOR_FORCE_FRONT_MIN = 0.45
CORRIDOR_HOLD_DIAGONAL_MIN = 0.40
 
##Lectura de API_KEY
with open('api_keys.txt', 'r') as f:
    content = f.readline()
API_KEY = content.split('=')[1]

##Declaracion del robot
#robot = Robot()
robot = Supervisor()
node = robot.getFromDef('PIONEER')

""" Inicializacion de sensores y actuadores """
## Ruedas
front_left_wheel = robot.getDevice('front left wheel')
front_right_wheel = robot.getDevice('front right wheel')
back_left_wheel = robot.getDevice('back left wheel')
back_right_wheel = robot.getDevice('back right wheel')
for wheel in [front_left_wheel, front_right_wheel, back_left_wheel, back_right_wheel]:
    wheel.setPosition(float('inf'))
    wheel.setVelocity(0)

## Lidar
lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)
resolution = lidar.getHorizontalResolution()
max_range = lidar.getMaxRange()

## Variables de navegacion
step_count = 0
state = 0
state_start_step = 0
front_velocity = np.random.uniform(0.5, 1)
wall_repulsion = 0.1
braintenberg_answer = lambda x, y: "RIGHT" if x > y else "LEFT"

recovery_steps = 0
recovery_turn_direction = 1
corridor_hold_until = 0

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
    if front < 0.60:
        return False

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

    return (side_pair and diagonal_pair) or funnel_entry or tight_entry or bottom_corridor_entry


def local_state_decision(left, left_front, front, right_front, right):
    mostly_open = min(left, left_front, front, right_front, right) > 1.35

    if is_corridor_candidate(left, left_front, front, right_front, right):
        return 1
    if mostly_open:
        return 2
    return 0


def validate_state_decision(selected_state, left, left_front, front, right_front, right):
    front_blocked = front < FRONT_DANGER or left_front < DIAGONAL_DANGER or right_front < DIAGONAL_DANGER
    side_too_close = left < SIDE_WARN or right < SIDE_WARN

    # Groq propone el comportamiento, pero estas reglas evitan decisiones fisicamente malas.
    if selected_state == 1 and not is_corridor_candidate(left, left_front, front, right_front, right):
        return 0
    if selected_state == 2 and (front_blocked or side_too_close):
        return 0

    return selected_state

# =========================
# Variables para mapa de calor
# =========================

pos_x = []
pos_y = []


## Asistente virtual LLM
LLM_assistant = LLM_decider.LLMDecider(API_KEY)

while robot.step(TIME_STEP) != -1 and step_count < MAX_STEPS:
    step_count += 1
    
    ## Se descompone las señales del Lidar para que sean mas faciles de utilizar
    scan = np.array(lidar.getRangeImage())
    scan[np.isinf(scan)] = max_range
    left = np.min(scan[85:110])
    left_front = np.min(scan[110:160])
    front = np.min(scan[160:200])
    right_front = np.min(scan[200:250])
    right = np.min(scan[250:275])
    
    ## Pre-procesamiento de sensores
    sensors = np.array([left, left_front, front, right_front, right])
    left_side = (left  * 0.6 + left_front * 0.4) / max_range
    right_side = (right * 0.6 + right_front * 0.4) / max_range
    free_space = (left_side - right_side) * 2
    front_cover = (front / max_range)
    front_blocked_now = front < FRONT_DANGER or left_front < DIAGONAL_DANGER or right_front < DIAGONAL_DANGER
    side_warning_now = left < SIDE_WARN or right < SIDE_WARN
    side_danger_now = left < SIDE_DANGER or right < SIDE_DANGER
    corridor_now = is_corridor_candidate(left, left_front, front, right_front, right)

    if corridor_now and state != 1 and recovery_steps == 0:
        state = 1
        state_start_step = step_count
        corridor_hold_until = step_count + CORRIDOR_HOLD_STEPS

    corridor_hold_valid = (
        front > CORRIDOR_FORCE_FRONT_MIN
        and min(left, right) >= SIDE_CRITICAL
        and min(left_front, right_front) >= CORRIDOR_HOLD_DIAGONAL_MIN
    )

    if state == 1 and not corridor_hold_valid:
        state = 0
        state_start_step = step_count
        corridor_hold_until = 0

    if state == 1 and step_count < corridor_hold_until and corridor_hold_valid:
        front_blocked_now = False
        side_warning_now = False

    lateral_only_now = front > FRONT_CLEAR and min(left_front, right_front) > 1.10

    if state == 2 and (front_blocked_now or (side_warning_now and not lateral_only_now)):
        state = 0
        state_start_step = step_count
    
    ## Maquina de estados
    if recovery_steps > 0:
        ## Maniobra de recuperacion: giro corto sobre su eje, sin alejarse demasiado.
        vel_left = 0.35 * recovery_turn_direction
        vel_right = -0.35 * recovery_turn_direction
        recovery_steps -= 1

    elif state == 0:
        ## Estado 0: Evasion de obstaculos
        if step_count % 500 == 0:
            front_velocity = np.random.uniform(0.5, 1)

        side_crash_now = min(left, right) < SIDE_DANGER and abs(left - right) > 0.25 and front < FRONT_CLEAR

        if front < FRONT_CRITICAL or side_crash_now:
            left_opening = left + left_front
            right_opening = right + right_front
            recovery_turn_direction = -1 if left_opening > right_opening else 1
            recovery_steps = RECOVERY_STEPS
            vel_left = 0.35 * recovery_turn_direction
            vel_right = -0.35 * recovery_turn_direction
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
            turn = np.clip(turn_error * ((1-front_cover) + wall_repulsion), -max_turn, max_turn)
            vel_left = front_velocity * front_cover - turn
            vel_right = front_velocity * front_cover + turn
    
    elif state == 1:
        ## Estado 1: Pasillo angosto
        # En pasillo prioriza centrarse: corrige hacia el lado con mas distancia lateral.
        center_error = (right - left) / max_range
        off_center = min(1.0, abs(center_error) * 2.0)
        corridor_width = min(1.0, (left + right) / (2.0 * CORRIDOR_MAX_SIDE))
        front_priority = np.clip(front / FRONT_CLEAR, 0.0, 1.0)
        forward = (0.45 + 0.18 * corridor_width) - 0.14 * off_center
        max_center_turn = 0.06 + 0.08 * (1.0 - front_priority)
        center_error = deadband(center_error, TURN_DEADBAND * 0.5)
        turn = np.clip(center_error * 0.42, -max_center_turn, max_center_turn)

        if front < 0.55:
            turn += np.clip((right_front - left_front) / max_range, -0.10, 0.10)
            forward = 0.18
        elif front < 0.90:
            turn += np.clip((right_front - left_front) / max_range, -0.08, 0.08)
            forward = min(forward, 0.32)

        vel_left = forward + turn
        vel_right = forward - turn
    
    elif state == 2:
        ## Estado 2: Exploracion
        # En espacios abiertos avanza recto; si aparece un obstaculo, corrige suavemente.
        obstacle_pressure = max(0.0, 1.0 - front_cover)
        sweep_turn = 0.0
        turn_error = deadband(-free_space, TURN_DEADBAND)
        avoid_turn = np.clip(turn_error * (obstacle_pressure + 0.18), -0.20, 0.20)
        forward = np.clip(0.75 * front_cover + 0.20, 0.25, 0.95)

        vel_left = forward + sweep_turn + avoid_turn
        vel_right = forward - sweep_turn - avoid_turn
    
    ## Modificacion de la velocidad de ruedas
    vel_left = np.clip(vel_left, -1, 1) * MAX_SPEED
    vel_right = np.clip(vel_right, -1, 1) * MAX_SPEED
    
    front_left_wheel.setVelocity(vel_left)
    back_left_wheel.setVelocity(vel_left)
    front_right_wheel.setVelocity(vel_right)
    back_right_wheel.setVelocity(vel_right)
    
    # =========================
    # Guardar posicion del robot
    # =========================

    position = node.getPosition()

    pos_x.append(position[0])
    pos_y.append(position[1])

    if step_count > STUCK_WINDOW and step_count % 40 == 0 and recovery_steps == 0:
        dx = pos_x[-1] - pos_x[-STUCK_WINDOW]
        dy = pos_y[-1] - pos_y[-STUCK_WINDOW]
        displacement = np.sqrt(dx * dx + dy * dy)

        if displacement < STUCK_DISTANCE:
            state = 0
            state_start_step = step_count

            if front > FRONT_CLEAR:
                recovery_steps = 0
                front_velocity = 0.85
            else:
                recovery_turn_direction = -1 if (left + left_front) > (right + right_front) else 1
                recovery_steps = STUCK_RECOVERY_STEPS

            print("Recuperacion por atasco: desplazamiento={:.3f}".format(displacement))
    
    # ## Envio mensaje a asistente
    front_blocked = front < FRONT_DANGER or left_front < DIAGONAL_DANGER or right_front < DIAGONAL_DANGER
    side_too_close = left < SIDE_WARN or right < SIDE_WARN
    corridor_hold_valid = (
        front > CORRIDOR_FORCE_FRONT_MIN
        and min(left, right) >= SIDE_CRITICAL
        and min(left_front, right_front) >= CORRIDOR_HOLD_DIAGONAL_MIN
    )
    decision_moment = step_count % DECISION_INTERVAL == 0

    # Reaccion local rapida: no espera a Groq si el robot queda mirando una pared.
    # En pasillo se permite cercania lateral moderada para que el estado 1 pueda centrarse.
    if front_blocked and state != 0 and not (
        state == 1
        and step_count < corridor_hold_until
        and corridor_hold_valid
    ):
        state = 0
        state_start_step = step_count
    elif side_too_close and state == 2 and not (
        front > FRONT_CLEAR and min(left_front, right_front) > 1.10
    ):
        state = 0
        state_start_step = step_count

    if decision_moment and step_count - state_start_step >= 25:
        robot_state = {
            "state": state,
            "steps_in_state": step_count - state_start_step,
        }
        answer = LLM_assistant.decide(sensors, robot_state)
        selected_state = parse_state_answer(answer)

        if selected_state is None:
            selected_state = local_state_decision(left, left_front, front, right_front, right)
        else:
            selected_state = validate_state_decision(selected_state, left, left_front, front, right_front, right)

        if (
            state == 1
            and step_count < corridor_hold_until
            and corridor_hold_valid
        ):
            selected_state = 1

        if selected_state != state:
            state = selected_state
            state_start_step = step_count
            if state == 1:
                corridor_hold_until = step_count + CORRIDOR_HOLD_STEPS

        if answer is None:
            print("Ningun mensaje de Groq, decision local aplicada")
        else:
            answer = answer.strip().upper()

            print("Respuesta Braitenberg: {}".format(braintenberg_answer(vel_left, vel_right)))
            print("Respuesta Groq: {}".format(answer))
            print("Estado aplicado: {} - {}".format(state, STATE_NAMES[state]))
            print("Estado sensores: {}".format(np.round(sensors, 2)))
            print("Lejania izquierda: {:.2f}".format(left_side * max_range))
            print("Lejania derecha: {:.2f}".format(right_side * max_range))
            print("--------------------------------------")
    
# =========================
# Mapa de calor
# =========================

## Frenar el robot al terminar
front_left_wheel.setVelocity(0)
back_left_wheel.setVelocity(0)
front_right_wheel.setVelocity(0)
back_right_wheel.setVelocity(0)


## Mapa de calor
pos_x = np.array(pos_x)
pos_y = np.array(pos_y)

heat_map = binned_statistic_2d(
    pos_x,
    pos_y,
    np.zeros(pos_x.shape),
    statistic='count',
    bins=20,
    range=[[-5, 5], [-5, 5]]
)

plt.figure()
plt.imshow(
    np.transpose(heat_map.statistic),
    origin='lower'
)

plt.colorbar()
plt.title("Mapa de calor del recorrido del robot")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()

heatmap_path = os.path.abspath("heatmap_recorrido_13.png")
plt.savefig(heatmap_path, dpi=150)
print("Mapa de calor guardado en: {}".format(heatmap_path))

try:
    if os.name == "nt":
        os.startfile(heatmap_path)
except Exception as error:
    print("No se pudo abrir automaticamente el mapa: {}".format(error))

plt.show(block=True)