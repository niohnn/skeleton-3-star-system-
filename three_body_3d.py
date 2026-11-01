
import glfw
import moderngl
import numpy as np
import math
import time
import threading
import dearpygui.dearpygui as dpg

# ============================================================
# SETTINGS
# ============================================================

WIDTH, HEIGHT = 1400, 900
DT = 0.00001
STEPS_PER_FRAME = 10
TRAIL_LENGTH = 1800
TRAIL_UPDATE_INTERVAL = 0.10

G = 4.0 * math.pi**2  # AU^3 / (solar mass * year^2)
C_AU_PER_YEAR = 63241.077
# Simulation controls
paused = False
time_scale = 1.0
star_scale = 1.0
planet_scale = 1.0
show_planets = True
show_trails = True

# ============================================================
# SHADERS
# ============================================================

VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_color;
in float in_size;
uniform mat4 mvp;
out vec3 vertex_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    gl_PointSize = in_size;
    vertex_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 vertex_color;
out vec4 frag_color;
void main() {
    vec2 p = gl_PointCoord - vec2(0.5);
    if (dot(p, p) > 0.25) discard;
    frag_color = vec4(vertex_color, 1.0);
}
"""

LINE_VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_color;
uniform mat4 mvp;
out vec3 vertex_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    vertex_color = in_color;
}
"""

LINE_FRAGMENT_SHADER = """
#version 330
in vec3 vertex_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(vertex_color, 1.0);
}
"""
BODY_VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_color;
in float in_size;

uniform mat4 view;
uniform mat4 projection;

out vec3 vertex_color;
flat out vec3 view_center;
flat out float point_size;

void main() {
    vec4 center = view * vec4(in_position, 1.0);

    view_center = center.xyz;
    vertex_color = in_color;
    point_size = in_size;

    gl_Position = projection * center;
    gl_PointSize = in_size;
}
"""

BODY_FRAGMENT_SHADER = """
#version 330

in vec3 vertex_color;
flat in vec3 view_center;
flat in float point_size;

uniform mat4 projection;
uniform float viewport_height;

out vec4 frag_color;

void main() {
    vec2 p = (gl_PointCoord - vec2(0.5)) * 2.0;

    float r2 = dot(p, p);

    if (r2 > 1.0) {
        discard;
    }

    float depth = -view_center.z;

    float radius = point_size * depth /
                   (projection[1][1] * viewport_height);

    float z = sqrt(1.0 - r2);

    vec3 surface_position = view_center +
        vec3(p.x * radius, p.y * radius, z * radius);

    vec4 clip_position = projection * vec4(surface_position, 1.0);

    gl_FragDepth = (clip_position.z / clip_position.w) * 0.5 + 0.5;

    frag_color = vec4(vertex_color, 1.0);
}
"""


# ============================================================
# THREE STARS
# ============================================================

star_masses = np.array([1.0, 0.8, 1.2], dtype=np.float64)
star_densities = np.array([
    1.41,  # Star 1
    1.41,  # Star 2
    1.41,  # Star 3
], dtype=np.float64)

star_positions = np.array([
    [-0.8, 0.0, 0.3],
    [ 0.8, -0.3, -0.2],
    [ 0.1, 1.0, 0.7],
], dtype=np.float64)

star_velocities = np.array([
    [-1.5, -2.5,  1.2],
    [ 2.5,  1.0, -1.5],
    [-1.0,  0.8,  1.0],
], dtype=np.float64)

# ============================================================
# SIX ROCKY PLANETS + FOUR GAS GIANTS
# Mass: solar masses
# Density: g/cm^3
# Orbital radius: AU
# ============================================================

planet_names = [
    "Rock 1", "Rock 2", "Rock 3", "Rock 4", "Rock 5", "Rock 6",
    "Gas 1", "Gas 2", "Gas 3", "Gas 4",
]

planet_types = ["rocky"] * 6 + ["gas"] * 4

planet_masses = np.array([
    1.0e-6, 2.0e-6, 3.0e-6, 2.5e-6, 4.0e-6, 5.0e-6,
    1.0e-4, 2.0e-4, 3.0e-4, 5.0e-4,
], dtype=np.float64)

planet_densities = np.array([
    5.5, 5.2, 4.5, 5.0, 3.9, 4.0,
    1.3, 0.8, 0.5, 0.3,
], dtype=np.float64)

planet_orbit_radii = np.array([
    0.12, 0.18, 0.25, 0.34, 0.44, 0.56,
    0.72, 0.88, 1.05, 1.22,
], dtype=np.float64)

planet_colors = [
    (0.75, 0.67, 0.57),
    (0.75, 0.35, 0.20),
    (0.80, 0.72, 0.48),
    (0.55, 0.52, 0.48),
    (0.65, 0.38, 0.28),
    (0.68, 0.58, 0.42),
    (0.85, 0.63, 0.40),
    (0.75, 0.67, 0.50),
    (0.70, 0.78, 0.85),
    (0.85, 0.72, 0.58),
]

NUM_STARS = len(star_masses)
NUM_PLANETS = len(planet_masses)
NUM_BODIES = NUM_STARS + NUM_PLANETS

# ============================================================
# PLANET INITIAL CONDITIONS
# Approximate circular orbits around star 0.
# Other stars will perturb the orbits.
# ============================================================

planet_positions = []
planet_velocities = []

for i, radius in enumerate(planet_orbit_radii):
    angle = i * 2.399963229728653
    inclination = np.radians((i - 4.5) * 5.0)

    radial = np.array([
        np.cos(angle),
        np.sin(angle) * np.cos(inclination),
        np.sin(angle) * np.sin(inclination),
    ])

    tangent = np.array([
        -np.sin(angle),
        np.cos(angle) * np.cos(inclination),
        np.cos(angle) * np.sin(inclination),
    ])

    position = star_positions[0] + radius * radial

    speed = np.sqrt(G * star_masses[0] / radius)
    velocity = star_velocities[0] + speed * tangent

    planet_positions.append(position)
    planet_velocities.append(velocity)

planet_positions = np.array(planet_positions, dtype=np.float64)
planet_velocities = np.array(planet_velocities, dtype=np.float64)

# Combine the stars and planets into one gravitational system.
masses = np.concatenate([star_masses, planet_masses])
positions = np.vstack([star_positions, planet_positions])
velocities = np.vstack([star_velocities, planet_velocities])

# Center-of-mass velocity correction.
velocities -= (
    np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)
)

# ============================================================
# DISPLAY SETTINGS
# Point sizes are pixels, not physical radii.
# Planets are exaggerated so they can be seen.
# ============================================================

star_colors = [
    (1.0, 0.82, 0.48),
    (0.55, 0.72, 1.0),
    (1.0, 0.45, 0.28),
]

body_colors = star_colors + planet_colors

body_sizes = np.array([
    17, 15, 19,              # stars
    5, 5, 6, 6, 6, 7,        # rocky planets
    9, 10, 11, 12,            # gas giants
], dtype=np.float32)
#save originals for reset
initial_positions = positions.copy()
initial_velocities = velocities.copy()
initial_masses = masses.copy()
initial_body_sizes = body_sizes.copy()

# Calculate physical planet radii from mass and density.
SOLAR_MASS_KG = 1.98847e30
AU_METERS = 1.495978707e11

planet_radii_au = []

for mass, density in zip(planet_masses, planet_densities):
    mass_kg = mass * SOLAR_MASS_KG
    density_kg_m3 = density * 1000.0
    radius_m = (
        3.0 * mass_kg / (4.0 * math.pi * density_kg_m3)
    ) ** (1.0 / 3.0)
    planet_radii_au.append(radius_m / AU_METERS)

planet_radii_au = np.array(planet_radii_au)

star_radii_au = []

for mass, density in zip(star_masses, star_densities):
    mass_kg = mass * SOLAR_MASS_KG
    density_kg_m3 = density * 1000.0

    radius_m = (
        3.0 * mass_kg / (4.0 * math.pi * density_kg_m3)
    ) ** (1.0 / 3.0)

    star_radii_au.append(radius_m / AU_METERS)

star_radii_au = np.array(star_radii_au)

collision_radii = np.concatenate([
    star_radii_au,
    planet_radii_au
])



# ============================================================
# GRAVITY FOR ALL 13 BODIES
# ============================================================

def acceleration(pos):
    # delta[i, j] points from body i toward body j.
    delta = pos[None, :, :] - pos[:, None, :]
    distance_sq = np.sum(delta * delta, axis=2)

    np.fill_diagonal(distance_sq, np.inf)
    distance_sq = np.maximum(distance_sq, 1e-12)

    inv_distance_cubed = distance_sq ** (-1.5)

    return G * np.sum(
        delta
        * (masses[None, :, None] * inv_distance_cubed[:, :, None]),
        axis=1,
    )


def physics_step():
    # Velocity-Verlet / leapfrog-style integration.
    global positions, velocities, paused

    a0 = acceleration(positions)
    velocities += 0.5 * a0 * DT
    positions += velocities * DT

    # Check for collisions at the new positions.
    collision = detect_collision(positions)

    if collision is not None:
        body_a, body_b = collision
        print(f"COLLISION! Body {body_a} and Body {body_b}")
        paused = True
        return

    a1 = acceleration(positions)
    velocities += 0.5 * a1 * DT

#collision detection
def detect_collision(pos):
    for i in range(NUM_BODIES):
        for j in range(i + 1, NUM_BODIES):
            distance = np.linalg.norm(pos[i] - pos[j])

            minimum_distance = collision_radii[i] + collision_radii[j]

            if distance <= minimum_distance:
                return i, j

    return None    

#body controls init for gui

def set_body_value(sender, app_data, user_data):
    property_name, body_index, component = user_data

    if property_name == "position":
        positions[body_index, component] = app_data
        history[body_index].clear()
        history[body_index].append(
            (simulation_years, positions[body_index].copy())
        )

    elif property_name == "velocity":
        velocities[body_index, component] = app_data

    elif property_name == "mass":
        masses[body_index] = app_data

        for i in range(NUM_BODIES):
            history[i].clear()
            history[i].append(
                (simulation_years, positions[i].copy())
            )

    elif property_name == "size":
        body_sizes[body_index] = app_data

#reset
def reset_simulation():
    global simulation_years, step_accumulator, paused
    global last_trail_update

    positions[:] = initial_positions
    velocities[:] = initial_velocities
    masses[:] = initial_masses
    body_sizes[:] = initial_body_sizes

    simulation_years = 0.0
    step_accumulator = 0.0
    paused = False

    for i in range(NUM_BODIES):
        history[i].clear()

        # Two identical starting points make the trajectory visible immediately.
        start_position = positions[i].copy()

        history[i].append((simulation_years, start_position.copy()))
        history[i].append((simulation_years + DT, start_position.copy()))

    update_trails()
    last_trail_update = time.time()      

def add_body_controls(body_index, body_name):
    with dpg.collapsing_header(label=body_name, default_open=False):

        dpg.add_text("Position (AU)")

        for axis, component in zip(["X", "Y", "Z"], range(3)):
            dpg.add_input_float(
                label=f"Position {axis}",
                default_value=float(positions[body_index, component]),
                callback=set_body_value,
                user_data=("position", body_index, component),
                width=200
            )

        dpg.add_separator()
        dpg.add_text("Velocity (AU/year)")

        for axis, component in zip(["X", "Y", "Z"], range(3)):
            dpg.add_input_float(
                label=f"Velocity {axis}",
                default_value=float(velocities[body_index, component]),
                callback=set_body_value,
                user_data=("velocity", body_index, component),
                width=200
            )

        dpg.add_separator()

        dpg.add_input_float(
            label="Mass (solar masses)",
            default_value=float(masses[body_index]),
            callback=set_body_value,
            user_data=("mass", body_index, None),
            width=200
        )

        dpg.add_input_float(
            label="Visual Size",
            default_value=float(body_sizes[body_index]),
            callback=set_body_value,
            user_data=("size", body_index, None),
            width=200
        )

       


# ============================================================
# LIGHT-TRAVEL-TIME HISTORY FOR THE THREE STARS
# ============================================================

OBSERVER = np.array([0.0, 0.0, -100.0], dtype=np.float64)



def apparent_position(body_index, current_time, body_history):
    """Approximate retarded position for a star."""
    current_position = positions[body_index]
    distance = np.linalg.norm(current_position - OBSERVER)
    delay = distance / C_AU_PER_YEAR

    history = body_history[body_index]
    if len(history) < 2:
        return current_position

    target_time = current_time - delay
    history_times = np.array([item[0] for item in history])
    history_positions = np.array([item[1] for item in history])

    if target_time <= history_times[0]:
        return history_positions[0]

    if target_time >= history_times[-1]:
        return current_position

    result = np.array([
        np.interp(target_time, history_times, history_positions[:, axis])
        for axis in range(3)
    ])
    return result


# Store simulation time and positions for the light-delay calculation.
history = [[] for _ in range(NUM_BODIES)]
simulation_years = 0.0

for i in range(NUM_BODIES):
    history[i].append((simulation_years, positions[i].copy()))

# ============================================================
# BACKGROUND STARS
# ============================================================

NUM_BACKGROUND_STARS = 10000
rng = np.random.default_rng(42)

directions = rng.normal(size=(NUM_BACKGROUND_STARS, 3))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)

visual_distances = rng.uniform(3.5, 8.0, size=(NUM_BACKGROUND_STARS, 1))
background_positions = directions * visual_distances

temperatures = rng.uniform(0.0, 1.0, NUM_BACKGROUND_STARS)
background_colors = np.zeros((NUM_BACKGROUND_STARS, 3), dtype=np.float32)

for i, t in enumerate(temperatures):
    if t < 0.33:
        background_colors[i] = (0.55, 0.68, 1.0)
    elif t < 0.66:
        background_colors[i] = (1.0, 1.0, 0.90)
    else:
        background_colors[i] = (1.0, 0.66, 0.40)

background_sizes = rng.uniform(
    1.0, 2.5, NUM_BACKGROUND_STARS
).astype(np.float32)

background_data = np.hstack([
    background_positions.astype(np.float32),
    background_colors,
    background_sizes[:, None],
]).astype(np.float32)

# ============================================================
# CAMERA MATRIX HELPERS
# ============================================================

def perspective(fov_degrees, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far),
         (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def look_at(eye, target, up):
    forward = target - eye
    forward /= np.linalg.norm(forward)

    side = np.cross(forward, up)
    side /= np.linalg.norm(side)

    up_vector = np.cross(side, forward)

    return np.array([
        [side[0], side[1], side[2], -np.dot(side, eye)],
        [up_vector[0], up_vector[1], up_vector[2],
         -np.dot(up_vector, eye)],
        [-forward[0], -forward[1], -forward[2],
         np.dot(forward, eye)],
        [0, 0, 0, 1],
    ], dtype=np.float32)


# ============================================================
# OPENGL INITIALIZATION
# ============================================================

if not glfw.init():
    raise RuntimeError("Could not initialize GLFW.")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

window = glfw.create_window(
    WIDTH, HEIGHT, "Three Stars + Ten Planets", None, None
)

if not window:
    glfw.terminate()
    raise RuntimeError("Could not create OpenGL window.")

glfw.make_context_current(window)
glfw.swap_interval(1)

ctx = moderngl.create_context()
ctx.enable(moderngl.PROGRAM_POINT_SIZE)
ctx.enable(moderngl.DEPTH_TEST)

point_program = ctx.program(
    vertex_shader=VERTEX_SHADER,
    fragment_shader=FRAGMENT_SHADER,
)

body_program = ctx.program(
    vertex_shader=BODY_VERTEX_SHADER,
    fragment_shader=BODY_FRAGMENT_SHADER,
)

line_program = ctx.program(
    vertex_shader=LINE_VERTEX_SHADER,
    fragment_shader=LINE_FRAGMENT_SHADER,
)

print("OpenGL renderer:", ctx.info.get("GL_RENDERER", "Unknown"))
print("Bodies: 3 stars, 6 rocky planets, 4 gas giants")
print("Drag mouse to rotate; scroll to zoom; ESC to exit.")

# Background star GPU buffers.
background_buffer = ctx.buffer(background_data.tobytes())
background_vao = ctx.vertex_array(
    point_program,
    [(background_buffer, "3f 3f 1f",
      "in_position", "in_color", "in_size")],
)

# Dynamic body buffer.
body_buffer = ctx.buffer(reserve=NUM_BODIES * 7 * 4)
body_vao = ctx.vertex_array(
    body_program,
    [(body_buffer, "3f 3f 1f",
      "in_position", "in_color", "in_size")],
)

# ============================================================
# TRAJECTORY RENDERING
# ============================================================

trail_vaos = [None] * NUM_BODIES
trail_buffers = [None] * NUM_BODIES
last_trail_update = 0.0

def update_trails():
    global trail_vaos, trail_buffers

    for i in range(NUM_BODIES):
        if trail_vaos[i] is not None:
            trail_vaos[i].release()
            trail_vaos[i] = None

        if trail_buffers[i] is not None:
            trail_buffers[i].release()
            trail_buffers[i] = None

        if len(history[i]) < 2:
            continue

        points = np.array(
            [entry[1] for entry in history[i]],
            dtype=np.float32,
        )

        color = np.array(body_colors[i], dtype=np.float32)
        colors = np.tile(color, (len(points), 1))
        data = np.hstack([points, colors]).astype(np.float32)

        trail_buffers[i] = ctx.buffer(data.tobytes())
        trail_vaos[i] = ctx.vertex_array(
            line_program,
            [(trail_buffers[i], "3f 3f",
              "in_position", "in_color")],
        )



# ============================================================
# FREE-FLY CAMERA CONTROLS
# ============================================================

camera_position = np.array([0.0, 0.0, 4.5], dtype=np.float64)

camera_yaw = -math.pi / 2
camera_pitch = 0.0

camera_speed = 2.0
mouse_sensitivity = 0.003

mouse_down = False
last_mouse_x = 0.0
last_mouse_y = 0.0


def get_camera_forward():
    return np.array([
        math.cos(camera_yaw) * math.cos(camera_pitch),
        math.sin(camera_pitch),
        math.sin(camera_yaw) * math.cos(camera_pitch),
    ], dtype=np.float64)


def mouse_button_callback(window, button, action, mods):
    global mouse_down, last_mouse_x, last_mouse_y

    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_down = action == glfw.PRESS
        last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)


def cursor_position_callback(window, x, y):
    global camera_yaw, camera_pitch
    global last_mouse_x, last_mouse_y

    if mouse_down:
        camera_yaw += (x - last_mouse_x) * mouse_sensitivity
        camera_pitch -= (y - last_mouse_y) * mouse_sensitivity

        camera_pitch = float(np.clip(
            camera_pitch, -1.5, 1.5
        ))

    last_mouse_x = x
    last_mouse_y = y


def scroll_callback(window, x_offset, y_offset):
    global camera_position

    forward = get_camera_forward()
    camera_position += forward * y_offset * 0.5


def update_camera(delta_time):
    global camera_position

    forward = get_camera_forward()

    world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)

    movement = np.zeros(3, dtype=np.float64)

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        movement += forward

    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        movement -= forward

    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        movement += right

    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        movement -= right

    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
        movement += world_up

    if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
        movement -= world_up

    if np.linalg.norm(movement) > 0:
        movement /= np.linalg.norm(movement)
        camera_position += movement * camera_speed * delta_time


glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_position_callback)
glfw.set_scroll_callback(window, scroll_callback)
glfw.set_scroll_callback(window, scroll_callback)

# ============================================================
# control pannel
# ============================================================
def run_control_panel():
    global paused, time_scale

    dpg.create_context()

    with dpg.window(label="Simulation Controls", width=450, height=800):

        def toggle_pause():
            global paused
            paused = not paused
            dpg.set_item_label(
                "pause_button",
                "Resume" if paused else "Pause"
            )
        dpg.add_button(
            label="Reset Simulation",
            callback=reset_simulation
        )   

        dpg.add_button(
            label="Pause",
            tag="pause_button",
            callback=toggle_pause
        )

        dpg.add_slider_float(
            label="Time Speed",
            min_value=0.1,
            max_value=10.0,
            default_value=1.0,
            callback=lambda s, a: set_time_scale(a)
        )

        dpg.add_checkbox(
            label="Show Trajectories",
            default_value=True,
            callback=lambda s, a: set_show_trails(a)
        )

        dpg.add_separator()

        dpg.add_text("STAR SYSTEM")

        for i in range(NUM_STARS):
            add_body_controls(i, f"Star {i + 1}")

        dpg.add_separator()

        dpg.add_text("PLANETS")

        for i in range(NUM_STARS, NUM_BODIES):
            add_body_controls(i, f"Planet {i - NUM_STARS + 1}")

    dpg.create_viewport(
        title="Simulation Controls",
        width=470,
        height=850
    )

    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def set_time_scale(value):
    global time_scale
    time_scale = value


def set_star_scale(value):
    global star_scale
    star_scale = value


def set_planet_scale(value):
    global planet_scale
    planet_scale = value


def set_show_planets(value):
    global show_planets
    show_planets = value


def set_show_trails(value):
    global show_trails
    show_trails = value


#============================================================
#console thread
#============================================================
control_thread = threading.Thread(
    target=run_control_panel,
    daemon=True
)

control_thread.start()
#frame pass
last_frame_time = glfw.get_time()
# ============================================================
# MAIN LOOP
# ============================================================
try:
    while not glfw.window_should_close(window):
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # ==========================================
        # ADVANCE ALL 13 BODIES
        # ==========================================

        if not paused:

            steps = max(1, int(STEPS_PER_FRAME * time_scale))

            for _ in range(steps):

                # Advance physics by one timestep
                physics_step()
                
       
                simulation_years += DT

                # Save trajectory history
                for i in range(NUM_BODIES):
                    history[i].append(
                        (simulation_years, positions[i].copy())
                    )

                    if len(history[i]) > TRAIL_LENGTH:
                        history[i].pop(0)

        # ==========================================#
        # REFRESH TRAJECTORY BUFFERS
        # ==========================================#


        now = time.time()

        if now - last_trail_update >= TRAIL_UPDATE_INTERVAL:
            update_trails()
            last_trail_update = now

        # ==========================================
        # LIGHT-TRAVEL-TIME POSITIONS
        # ==========================================

        displayed_positions = positions.copy()

        for i in range(NUM_STARS):
            displayed_positions[i] = apparent_position(
                i, simulation_years, history
            )

        body_data = np.hstack([
            displayed_positions.astype(np.float32),
            np.array(body_colors, dtype=np.float32),
            body_sizes[:, None],
        ]).astype(np.float32)

        body_buffer.write(body_data.tobytes())
        current_time = glfw.get_time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        update_camera(delta_time)

        # CAMERA
        

        forward = get_camera_forward()

        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        width, height = glfw.get_framebuffer_size(window)
        width = max(width, 1)
        height = max(height, 1)

        projection = perspective(60.0, width / height, 0.01, 100.0)

        view = look_at(
             camera_position,
             camera_position + forward,
             up
             )

        mvp = projection @ view

        body_program["view"].write(view.T.astype("f4").tobytes())
        body_program["projection"].write(projection.T.astype("f4").tobytes())
        body_program["viewport_height"].value = float(height)

        point_program["mvp"].write(mvp.T.astype("f4").tobytes())
        line_program["mvp"].write(mvp.T.astype("f4").tobytes())

        ctx.viewport = (0, 0, width, height)
        ctx.clear(0.005, 0.007, 0.02, 1.0)

        # Background
        background_vao.render(
              mode=moderngl.POINTS,
              vertices=NUM_BACKGROUND_STARS,
            )
        

        # Trajectory trails
        for vao in trail_vaos:
            if vao is not None:
                vao.render(mode=moderngl.LINE_STRIP)

        # Main bodies
        body_vao.render(
            mode=moderngl.POINTS,
            vertices=NUM_BODIES,
        )

        glfw.swap_buffers(window)

        glfw.set_window_title(
            window,
            f"Three Stars + Ten Planets | {simulation_years:.5f} years"
        )

finally:
    for vao in trail_vaos:
        if vao is not None:
            vao.release()

    for buf in trail_buffers:
        if buf is not None:
            buf.release()

    background_vao.release()
    background_buffer.release()
    body_vao.release()
    body_buffer.release()
    point_program.release()
    line_program.release()
    body_program.release()
    ctx.release()


    glfw.destroy_window(window)
    glfw.terminate()
