from controller import Robot, Supervisor
import numpy as np
import LLM_decider
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

##Parametros de simulacion
TIME_STEP = 64
MAX_SPEED = 6.0
MAX_STEPS = 10000
 
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
front_velocity = np.random.uniform(0.5, 1)
wall_repulsion = 0.1
braintenberg_answer = lambda x, y: "RIGHT" if x > y else "LEFT"

# =========================
# Influencia de Groq
# =========================

ai_direction = "NONE"

# Fuerza con la que Groq influye en el giro.
# Probar valores entre 0.10 y 0.40.
AI_TURN_STRENGTH = 0.25

# La influencia se va apagando de a poco para que no quede girando eternamente.
AI_TURN_DECAY = 0.92

# Sesgo de giro actual.
# positivo => gira a la derecha
# negativo => gira a la izquierda
ai_turn_bias = 0.0

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
    
    ## Maquina de estados
    if state == 0:
        ## Estado 0: Evasion de obstaculos
        if step_count % 500 == 0:
            front_velocity = np.random.uniform(0.5, 1)
        vel_left = front_velocity * front_cover - free_space * ((1-front_cover) + wall_repulsion)
        vel_right = front_velocity * front_cover + free_space * ((1-front_cover) + wall_repulsion)
    
    if state == 1:
        ## TODO: Agregar otro comportamiento
        pass
    
    if state == 2:
        ## TODO: Agregar otro comportamiento
        pass
    
    # =========================
    # Influencia de Groq sobre el giro
    # =========================
    # ai_turn_bias > 0 => giro a la derecha
    # ai_turn_bias < 0 => giro a la izquierda
    
    vel_left = vel_left + ai_turn_bias
    vel_right = vel_right - ai_turn_bias
    
    # La influencia se va reduciendo con el tiempo
    ai_turn_bias = ai_turn_bias * AI_TURN_DECAY
    
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
    
    # ## Envio mensaje a asistente
    if step_count % 100 == 0:
        answer = LLM_assistant.decide(sensors)
        if answer is None:
            print("Ningun mensaje de Groq")
        else:
            answer = answer.strip().upper()
        
            if "RIGHT" in answer:
                ai_direction = "RIGHT"
                ai_turn_bias = AI_TURN_STRENGTH
        
            elif "LEFT" in answer:
                ai_direction = "LEFT"
                ai_turn_bias = -AI_TURN_STRENGTH
        
            print("Respuesta Braitenberg: {}".format(braintenberg_answer(vel_left, vel_right)))
            print("Respuesta Groq: {}".format(answer))
            print("Direccion IA aplicada: {}".format(ai_direction))
            print("Sesgo IA: {:.2f}".format(ai_turn_bias))
            print("Estado sensores: {}".format(np.round(sensors, 2)))
            print("Lejania izquierda: {:.2f}".format(left_side * max_range))
            print("Lejania derecha: {:.2f}".format(right_side * max_range))
            print("--------------------------------------")
    
    # ## Envio mensaje a asistente
    # if step_count % 100 == 0:
        # answer = LLM_assistant.decide(sensors)
        # if answer is None:
            # print("Ningun mensaje de Groq")
        # else:
            
            # print("Respuesta Braitenberg: {}".format(braintenberg_answer(vel_left, vel_right)))
            # print("Respuesta Groq: {}".format(answer))
            # print("Estado sensores: {}".format(np.round(sensors, 2)))
            # print("Lejania izquierda: {:.2f}".format(left_side * max_range))
            # print("Lejania derecha: {:.2f}".format(right_side * max_range))
            # print("--------------------------------------")
            
    

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
plt.show()
