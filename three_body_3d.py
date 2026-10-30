
import glfw
import moderngl
import numpy as np
import math
import time

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

# ============================================================
# THREE STARS
# ============================================================

star_masses = np.array([1.0, 0.8, 1.2], dtype=np.float64)

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
    global positions, velocities

    a0 = acceleration(positions)
    velocities += 0.5 * a0 * DT
    positions += velocities * DT
    a1 = acceleration(positions)
    velocities += 0.5 * a1 * DT


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
    point_program,
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
            trail_buffers[i].release()

        points = np.array(
            [entry[1] for entry in history[i]],
            dtype=np.float32,
        )

        if len(points) < 2:
            continue

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
# CAMERA CONTROLS
# ============================================================

camera_yaw = 0.6
camera_pitch = 0.35
camera_distance = 4.5

mouse_down = False
last_mouse_x = 0.0
last_mouse_y = 0.0


def mouse_button_callback(window, button, action, mods):
    global mouse_down, last_mouse_x, last_mouse_y

    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_down = action == glfw.PRESS
        last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)


def cursor_position_callback(window, x, y):
    global camera_yaw, camera_pitch, last_mouse_x, last_mouse_y

    if mouse_down:
        camera_yaw += (x - last_mouse_x) * 0.005
        camera_pitch += (y - last_mouse_y) * 0.005
        camera_pitch = float(np.clip(camera_pitch, -1.5, 1.5))

    last_mouse_x, last_mouse_y = x, y


def scroll_callback(window, x_offset, y_offset):
    global camera_distance
    camera_distance *= math.exp(-y_offset * 0.1)
    camera_distance = float(np.clip(camera_distance, 0.5, 30.0))


glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_position_callback)
glfw.set_scroll_callback(window, scroll_callback)

# ============================================================
# MAIN LOOP
# ============================================================

try:
    while not glfw.window_should_close(window):
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # Advance all 13 bodies using mutual gravity.
        for _ in range(STEPS_PER_FRAME):
            physics_step()
            simulation_years += DT

            for i in range(NUM_BODIES):
                history[i].append(
                    (simulation_years, positions[i].copy())
                )

                if len(history[i]) > TRAIL_LENGTH:
                    history[i].pop(0)

        # Refresh trajectory buffers periodically.
        now = time.time()
        if now - last_trail_update >= TRAIL_UPDATE_INTERVAL:
            update_trails()
            last_trail_update = now

        # Use light-travel-time positions for the three stars.
        # Planets use their current positions.
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

        # Camera.
        eye = np.array([
            camera_distance * math.cos(camera_pitch) * math.sin(camera_yaw),
            camera_distance * math.sin(camera_pitch),
            camera_distance * math.cos(camera_pitch) * math.cos(camera_yaw),
        ], dtype=np.float32)

        target = np.zeros(3, dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        width, height = glfw.get_framebuffer_size(window)
        width = max(width, 1)
        height = max(height, 1)

        projection = perspective(60.0, width / height, 0.01, 100.0)
        view = look_at(eye, target, up)
        mvp = projection @ view

        point_program["mvp"].write(mvp.T.astype("f4").tobytes())
        line_program["mvp"].write(mvp.T.astype("f4").tobytes())

        ctx.viewport = (0, 0, width, height)
        ctx.clear(0.005, 0.007, 0.02, 1.0)

        # Background.
        background_vao.render(
            mode=moderngl.POINTS,
            vertices=NUM_BACKGROUND_STARS,
        )

        # Trajectory trails for stars and planets.
        for vao in trail_vaos:
            if vao is not None:
                vao.render(mode=moderngl.LINE_STRIP)

        # Main bodies.
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
    ctx.release()

    glfw.destroy_window(window)
    glfw.terminate()
