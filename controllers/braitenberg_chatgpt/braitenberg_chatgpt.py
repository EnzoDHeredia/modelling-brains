from controller import Supervisor
import os
from datetime import datetime
import numpy as np
import LLM_decider
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

# =========================
# Parametros generales
# =========================
# TIME_STEP y MAX_STEPS definen la duracion de la simulacion.
# MAX_SPEED convierte velocidades normalizadas [-1, 1] a velocidad de ruedas Webots.
TIME_STEP = 64
MAX_SPEED = 6.0
MAX_STEPS = 10000

# Groq no se consulta en cada paso: se consulta cada DECISION_INTERVAL para evitar
# cambios bruscos y llamadas innecesarias al LLM.
DECISION_INTERVAL = 40

# Maniobra corta de giro cuando Braitenberg detecta una situacion critica.
RECOVERY_STEPS = 5

# Umbrales de distancia. Las lecturas vienen del LiDAR: valores bajos implican pared/obstaculo.
FRONT_DANGER = 0.90
FRONT_CRITICAL = 0.45
FRONT_CLEAR = 1.35
TURN_DEADBAND = 0.18

# Ganancias de giro. Estos valores controlan que las correcciones sean suaves.
RECOVERY_TURN_SPEED = 0.22
SIDE_AVOID_TURN_BASE = 0.025
SIDE_AVOID_TURN_GAIN = 0.06
FREE_AVOID_TURN_BASE = 0.05
FREE_AVOID_TURN_GAIN = 0.11
CORRIDOR_TURN_BASE = 0.04
CORRIDOR_TURN_GAIN = 0.05
CORRIDOR_CENTER_GAIN = 0.28
EXPLORATION_TURN_LIMIT = 0.12
EXPLORATION_RANDOM_TURN_MIN = 0.04
EXPLORATION_RANDOM_TURN_MAX = 0.12
EXPLORATION_TURN_STEPS = 120

# Umbrales laterales y diagonales usados para decidir si hay riesgo inmediato.
DIAGONAL_DANGER = 0.70
SIDE_WARN = 0.85
SIDE_DANGER = 0.35
SIDE_CRITICAL = 0.40

# Umbrales especificos de pasillo. Son mas permisivos que los de evasion porque
# en un pasillo angosto es normal tener paredes cerca a los lados.
CORRIDOR_MIN_SIDE = 0.30
CORRIDOR_MAX_SIDE = 1.35
CORRIDOR_MAX_IMBALANCE = 0.70
CORRIDOR_MAX_DIAGONAL_IMBALANCE = 0.80
CORRIDOR_FRONT_OPEN = 0.95
CORRIDOR_HOLD_STEPS = 180
CORRIDOR_FORCE_FRONT_MIN = 0.40
CORRIDOR_HOLD_DIAGONAL_MIN = 0.30
 
##Lectura de API_KEY
# La API key se mantiene fuera del codigo fuente. Si Groq falla,
# el controlador puede seguir funcionando con Braitenberg.
with open('api_keys.txt', 'r') as f:
    content = f.readline()
API_KEY = content.split('=')[1]

##Declaracion del robot
# Se usa Supervisor solo para registrar la posicion real en el heatmap.
# Esa posicion no participa en la navegacion ni en la deteccion de atascos.
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
# El LiDAR es la unica fuente de percepcion usada para decidir movimiento.
lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)
resolution = lidar.getHorizontalResolution()
max_range = lidar.getMaxRange()

## Variables de navegacion
# state es el comportamiento aplicado: 0=Braitenberg, 1=pasillo, 2=exploracion.
step_count = 0
state = 0
state_start_step = 0

# Braitenberg usa una velocidad frontal base que cambia ocasionalmente para
# evitar trayectorias completamente repetitivas.
front_velocity = np.random.uniform(0.5, 1)
wall_repulsion = 0.1

# Esta funcion no devuelve un estado; solo resume la accion de giro resultante.
braintenberg_answer = lambda x, y: "RIGHT" if x > y else "LEFT"

recovery_steps = 0
recovery_turn_direction = 1

# Mantiene temporalmente STATE_1 para que el robot no salga del modo pasillo por ruido.
corridor_hold_until = 0

# STATE_2 usa un giro aleatorio persistente durante varios pasos.
exploration_turn = 0.0
exploration_turn_until = 0

# Posiciones guardadas solo para construir el heatmap al final.
pos_x = []
pos_y = []

STATE_NAMES = {
    0: "Braitenberg/evasion",
    1: "Pasillo angosto",
    2: "Exploracion",
}


def deadband(value, threshold):
    """Elimina correcciones pequenas para que el robot no oscile por ruido."""
    if abs(value) < threshold:
        return 0.0

    return value


def parse_state_answer(answer):
    """Convierte la respuesta textual de Groq en un entero de estado."""
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
    """Detecta si las lecturas son compatibles con un pasillo angosto."""
    if front < CORRIDOR_FORCE_FRONT_MIN:
        return False

    # Pasillo clasico: paredes laterales detectadas, frente transitable y laterales parecidos.
    side_pair = (
        CORRIDOR_MIN_SIDE < left < 1.60
        and CORRIDOR_MIN_SIDE < right < 1.60
        and abs(left - right) < 0.95
    )
    # Las diagonales ayudan a descartar esquinas o choques inminentes.
    diagonal_pair = (
        left_front > CORRIDOR_HOLD_DIAGONAL_MIN
        and right_front > CORRIDOR_HOLD_DIAGONAL_MIN
        and abs(left_front - right_front) < 1.10
    )
    # Entradas alternativas para pasillos o embudos donde un lado aparece mas cerrado.
    funnel_entry = (
        front > 1.20
        and min(left, right) < 0.80
        and max(left, right) < 1.65
        and min(left_front, right_front) > 0.45
    )
    tight_entry = (
        front > CORRIDOR_FORCE_FRONT_MIN
        and min(left, right) > CORRIDOR_MIN_SIDE
        and max(left, right) > 1.00
        and max(left, right) < 1.70
        and min(left_front, right_front) > CORRIDOR_HOLD_DIAGONAL_MIN
    )
    bottom_corridor_entry = (
        front > CORRIDOR_FORCE_FRONT_MIN
        and min(left, right) > CORRIDOR_MIN_SIDE
        and max(left, right) > 0.90
        and max(left, right) < 1.70
        and min(left_front, right_front) > CORRIDOR_HOLD_DIAGONAL_MIN
    )

    return (side_pair and diagonal_pair) or funnel_entry or tight_entry or bottom_corridor_entry


def validate_state_decision(selected_state, left, left_front, front, right_front, right):
    """Valida que el estado propuesto por Groq sea fisicamente razonable."""
    front_blocked = front < FRONT_DANGER or left_front < DIAGONAL_DANGER or right_front < DIAGONAL_DANGER
    side_too_close = left < SIDE_WARN or right < SIDE_WARN
    mostly_open = min(left, left_front, front, right_front, right) > FRONT_CLEAR

    # Groq propone el comportamiento; si contradice los sensores, decide Braitenberg.
    # STATE_0 siempre es valido porque es el modo de seguridad/evasion.
    if selected_state == 0:
        return 0
    # STATE_1 solo se acepta si realmente hay geometria de pasillo.
    if selected_state == 1:
        if mostly_open or not is_corridor_candidate(left, left_front, front, right_front, right):
            return None
        return 1
    # STATE_2 solo se acepta si todo esta abierto y no hay riesgo cercano.
    if selected_state == 2:
        if not mostly_open or front_blocked or side_too_close:
            return None
        return 2

    return None

## Asistente virtual LLM
# Groq solo elige el estado; nunca calcula velocidades de ruedas.
LLM_assistant = LLM_decider.LLMDecider(API_KEY)

while robot.step(TIME_STEP) != -1 and step_count < MAX_STEPS:
    step_count += 1
    
    ## Se descompone las señales del Lidar para que sean mas faciles de utilizar
    # El escaneo completo se resume en cinco sectores: izquierda, front-left,
    # frente, front-right y derecha.
    scan = np.array(lidar.getRangeImage())
    scan[np.isinf(scan)] = max_range
    left = np.min(scan[85:110])
    left_front = np.min(scan[110:160])
    front = np.min(scan[160:200])
    right_front = np.min(scan[200:250])
    right = np.min(scan[250:275])
    
    ## Pre-procesamiento de sensores
    # left_side/right_side combinan lateral y diagonal para estimar que lado esta mas libre.
    sensors = np.array([left, left_front, front, right_front, right])
    left_side = (left  * 0.6 + left_front * 0.4) / max_range
    right_side = (right * 0.6 + right_front * 0.4) / max_range
    free_space = (left_side - right_side) * 2
    front_cover = (front / max_range)
    # Banderas de seguridad local: se evaluan antes de esperar una nueva respuesta de Groq.
    front_blocked_now = front < FRONT_DANGER or left_front < DIAGONAL_DANGER or right_front < DIAGONAL_DANGER
    side_warning_now = left < SIDE_WARN or right < SIDE_WARN
    side_danger_now = left < SIDE_DANGER or right < SIDE_DANGER

    # Condicion minima para sostener STATE_1 sin salir por una lectura lateral cercana normal.
    corridor_hold_valid = (
        front > CORRIDOR_FORCE_FRONT_MIN
        and min(left, right) >= CORRIDOR_MIN_SIDE
        and min(left_front, right_front) >= CORRIDOR_HOLD_DIAGONAL_MIN
    )

    if state == 1 and not corridor_hold_valid:
        state = 0
        state_start_step = step_count
        corridor_hold_until = 0

    # Mientras el hold del pasillo sea valido, se ignoran advertencias laterales moderadas.
    if state == 1 and step_count < corridor_hold_until and corridor_hold_valid:
        front_blocked_now = False
        side_warning_now = False

    # En exploracion, si aparece un obstaculo real, se vuelve inmediatamente a Braitenberg.
    lateral_only_now = front > FRONT_CLEAR and min(left_front, right_front) > 1.10

    if state == 2 and (front_blocked_now or (side_warning_now and not lateral_only_now)):
        state = 0
        state_start_step = step_count
    
    ## Maquina de estados
    if recovery_steps > 0:
        ## Maniobra de recuperacion: giro corto sobre su eje, sin alejarse demasiado.
        # Esta recuperacion se activa por sensores, no por posicion global del Supervisor.
        vel_left = RECOVERY_TURN_SPEED * recovery_turn_direction
        vel_right = -RECOVERY_TURN_SPEED * recovery_turn_direction
        recovery_steps -= 1

    elif state == 0:
        ## Estado 0: Evasion de obstaculos
        # Braitenberg es el modo de seguridad: reacciona solo con sensores locales.
        if step_count % 500 == 0:
            front_velocity = np.random.uniform(0.5, 1)

        side_crash_now = min(left, right) < SIDE_DANGER and abs(left - right) > 0.25 and front < FRONT_CLEAR

        # Si el frente esta critico, gira sobre su eje hacia el lado con mayor apertura.
        if front < FRONT_CRITICAL or side_crash_now:
            left_opening = left + left_front
            right_opening = right + right_front
            recovery_turn_direction = -1 if left_opening > right_opening else 1
            recovery_steps = RECOVERY_STEPS
            vel_left = RECOVERY_TURN_SPEED * recovery_turn_direction
            vel_right = -RECOVERY_TURN_SPEED * recovery_turn_direction
        elif side_warning_now:
            # Si hay una pared lateral cerca, corrige suavemente sin frenar por completo.
            left_clearance = 0.65 * left + 0.35 * left_front
            right_clearance = 0.65 * right + 0.35 * right_front
            front_priority = np.clip(front / FRONT_CLEAR, 0.0, 1.0)
            front_diagonal_clear = min(left_front, right_front) > 1.10
            lateral_only = front > FRONT_CLEAR and front_diagonal_clear

            max_turn = 0.01 if lateral_only else SIDE_AVOID_TURN_BASE + SIDE_AVOID_TURN_GAIN * (1.0 - front_priority)
            turn_error = deadband((right_clearance - left_clearance) / max_range, TURN_DEADBAND)
            turn = np.clip(turn_error, -max_turn, max_turn)
            forward = 0.78 if lateral_only else 0.62 + 0.16 * front_priority
            vel_left = forward + turn
            vel_right = forward - turn
        else:
            # Camino relativamente libre: avanza y aplica una correccion pequena hacia el espacio libre.
            front_priority = np.clip(front / FRONT_CLEAR, 0.0, 1.0)
            max_turn = FREE_AVOID_TURN_BASE + FREE_AVOID_TURN_GAIN * (1.0 - front_priority)
            turn_error = deadband(free_space, TURN_DEADBAND)
            turn = np.clip(turn_error * ((1-front_cover) + wall_repulsion), -max_turn, max_turn)
            vel_left = front_velocity * front_cover - turn
            vel_right = front_velocity * front_cover + turn
    
    elif state == 1:
        ## Estado 1: Pasillo angosto
        # En pasillo prioriza centrarse: corrige hacia el lado con mas distancia lateral.
        # Si right > left, el robot esta mas cerca de la pared izquierda y corrige a la derecha.
        center_error = (right - left) / max_range
        off_center = min(1.0, abs(center_error) * 2.0)
        corridor_width = min(1.0, (left + right) / (2.0 * CORRIDOR_MAX_SIDE))
        front_priority = np.clip(front / FRONT_CLEAR, 0.0, 1.0)
        forward = (0.45 + 0.18 * corridor_width) - 0.14 * off_center
        max_center_turn = CORRIDOR_TURN_BASE + CORRIDOR_TURN_GAIN * (1.0 - front_priority)
        center_error = deadband(center_error, TURN_DEADBAND * 0.5)
        turn = np.clip(center_error * CORRIDOR_CENTER_GAIN, -max_center_turn, max_center_turn)

        # Si el frente se estrecha, baja velocidad y usa diagonales para elegir una salida suave.
        if front < 0.55:
            turn += np.clip((right_front - left_front) / max_range, -0.06, 0.06)
            forward = 0.18
        elif front < 0.90:
            turn += np.clip((right_front - left_front) / max_range, -0.05, 0.05)
            forward = min(forward, 0.32)

        vel_left = forward + turn
        vel_right = forward - turn
    
    elif state == 2:
        ## Estado 2: Exploracion
        # En exploracion el giro es aleatorio; Braitenberg se activa cuando aparece riesgo.
        # La direccion se mantiene durante varios pasos para producir curvas, no zigzag por paso.
        if step_count >= exploration_turn_until:
            direction = np.random.choice([-1.0, 1.0])
            magnitude = np.random.uniform(EXPLORATION_RANDOM_TURN_MIN, EXPLORATION_RANDOM_TURN_MAX)
            exploration_turn = direction * magnitude
            exploration_turn_until = step_count + EXPLORATION_TURN_STEPS

        sweep_turn = np.clip(exploration_turn, -EXPLORATION_TURN_LIMIT, EXPLORATION_TURN_LIMIT)
        forward = np.clip(0.75 * front_cover + 0.20, 0.25, 0.95)

        vel_left = forward + sweep_turn
        vel_right = forward - sweep_turn
    
    ## Modificacion de la velocidad de ruedas
    # Todas las ramas anteriores producen velocidades normalizadas; aqui se escala a Webots.
    vel_left = np.clip(vel_left, -1, 1) * MAX_SPEED
    vel_right = np.clip(vel_right, -1, 1) * MAX_SPEED
    
    front_left_wheel.setVelocity(vel_left)
    back_left_wheel.setVelocity(vel_left)
    front_right_wheel.setVelocity(vel_right)
    back_right_wheel.setVelocity(vel_right)

    # Registro exclusivo para heatmap. No se usa para decidir estados ni detectar atascos.
    position = node.getPosition()
    pos_x.append(position[0])
    pos_y.append(position[1])
    
    # ## Envio mensaje a asistente
    # Se recalculan banderas para decidir si se consulta Groq y validar su propuesta.
    front_blocked = front < FRONT_DANGER or left_front < DIAGONAL_DANGER or right_front < DIAGONAL_DANGER
    side_too_close = left < SIDE_WARN or right < SIDE_WARN
    corridor_hold_valid = (
        front > CORRIDOR_FORCE_FRONT_MIN
        and min(left, right) >= CORRIDOR_MIN_SIDE
        and min(left_front, right_front) >= CORRIDOR_HOLD_DIAGONAL_MIN
    )
    decision_moment = step_count % DECISION_INTERVAL == 0

    # Reaccion local rapida: no espera a Groq si el robot queda mirando una pared.
    # En pasillo se permite cercania lateral moderada para que el estado 1 pueda centrarse.
    # Esto garantiza que la seguridad no dependa de la latencia del LLM.
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
        decision_source = "Groq"

        # Si Groq no responde o responde algo no parseable, se usa Braitenberg.
        if selected_state is None:
            selected_state = 0
            decision_source = "Braitenberg"
        else:
            # Groq puede sugerir estados, pero el controlador valida que sean seguros.
            selected_state = validate_state_decision(selected_state, left, left_front, front, right_front, right)
            if selected_state is None:
                selected_state = 0
                decision_source = "Braitenberg"

        # Cuando el estado aplicado es 0, la decision final corresponde a Braitenberg.
        if selected_state == 0:
            decision_source = "Braitenberg"

        if selected_state != state:
            previous_state = state
            state = selected_state
            state_start_step = step_count
            # Al entrar en pasillo se activa un hold para estabilizar el modo.
            if state == 1:
                corridor_hold_until = step_count + CORRIDOR_HOLD_STEPS
            else:
                corridor_hold_until = 0
            # Al entrar o salir de exploracion se reinicia el giro aleatorio.
            if state == 2 and previous_state != 2:
                exploration_turn_until = 0
            elif previous_state == 2 and state != 2:
                exploration_turn = 0.0
                exploration_turn_until = 0

        if answer is None:
            # El log separa propuesta de Groq, accion Braitenberg y estado aplicado.
            print("Paso: {}".format(step_count))
            print("Ningun mensaje de Groq, Braitenberg aplicado")
        else:
            answer = answer.strip().upper()

            print("Paso: {}".format(step_count))
            print("Respuesta Braitenberg: STATE_0 - {} (accion: {})".format(
                STATE_NAMES[0],
                braintenberg_answer(vel_left, vel_right)
            ))
            print("Respuesta Groq: {}".format(answer))
            print("Estado aplicado: {} - {}".format(state, STATE_NAMES[state]))
            print("Decision final: {}".format(decision_source))
            print("Estado sensores: {}".format(np.round(sensors, 2)))
            print("Lejania izquierda: {:.2f}".format(left_side * max_range))
            print("Lejania derecha: {:.2f}".format(right_side * max_range))
            print("--------------------------------------")
    
## Frenar el robot al terminar
# Al finalizar la simulacion se detienen todas las ruedas antes de generar graficas.
front_left_wheel.setVelocity(0)
back_left_wheel.setVelocity(0)
front_right_wheel.setVelocity(0)
back_right_wheel.setVelocity(0)

if pos_x and pos_y:
    # El titulo usa heatmap_HH:MM. El archivo usa HH-MM porque Windows no admite ':'.
    graph_name = datetime.now().strftime("heatmap_%H:%M")
    filename = "{}.png".format(graph_name.replace(":", "-"))
    heatmap_path = os.path.abspath(filename)

    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)

    # El heatmap cuenta cuantas muestras de posicion cayeron en cada celda.
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
    plt.title(graph_name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    print("Mapa de calor guardado en: {}".format(heatmap_path))
    plt.show(block=True)
