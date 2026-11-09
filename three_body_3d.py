
import glfw
import moderngl
import numpy as np
import math
import time
import threading
import dearpygui.dearpygui as dpg
from queue import SimpleQueue
from PIL import Image, ImageDraw, ImageFont
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
LABEL_VERTEX_SHADER = """
#version 330

uniform vec2 center_ndc;
uniform vec2 size_ndc;
out vec2 uv;

void main() {
    vec2 corners[4] = vec2[](
        vec2(-0.5, -0.5),
        vec2( 0.5, -0.5),
        vec2(-0.5,  0.5),
        vec2( 0.5,  0.5)
    );

    vec2 corner = corners[gl_VertexID];
    uv = vec2(corner.x + 0.5, 0.5 - corner.y);

    gl_Position = vec4(
        center_ndc + corner * size_ndc,
        0.0,
        1.0
    );
}
"""

LABEL_FRAGMENT_SHADER = """
#version 330

in vec2 uv;
uniform sampler2D label_texture;
out vec4 frag_color;

void main() {
    frag_color = texture(label_texture, uv);
}
"""
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
    vec2 p = (gl_PointCoord - vec2(0.5)) * 2.0;
    float radius = length(p);

    if (radius > 1.0) discard;

    float core = 1.0 - smoothstep(0.08, 0.45, radius);
    float glow = exp(-radius * radius * 5.0) * 0.35;
    float alpha = max(core, glow);

    frag_color = vec4(vertex_color * (core + glow), alpha);
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
NEBULA_VERTEX_SHADER = """
#version 330

out vec2 uv;

void main() {
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 position = positions[gl_VertexID];
    uv = position * 0.5 + 0.5;

    // Place the backdrop far behind the stars and bodies.
    gl_Position = vec4(position, 0.999, 1.0);
}
"""

NEBULA_FRAGMENT_SHADER = """
#version 330

uniform vec2 resolution;
uniform mat4 inverse_projection;
uniform mat4 inverse_view;
uniform vec3 camera_position;
uniform vec4 nebula_centers[32];
uniform vec4 nebula_colors[32];
out vec4 frag_color;

float hash(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = hash(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash(i + vec3(1.0, 1.0, 1.0));

    float x00 = mix(n000, n100, f.x);
    float x10 = mix(n010, n110, f.x);
    float x01 = mix(n001, n101, f.x);
    float x11 = mix(n011, n111, f.x);

    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

float fbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;

    for (int i = 0; i < 4; i++) {
        value += amplitude * noise(p);
        p = p * 2.0 + vec3(13.7, 9.2, 5.3);
        amplitude *= 0.5;
    }

    return value;
}

float nebula_blob(vec3 ray, vec4 center_radius, vec4 tint) {
    vec3 center = center_radius.xyz;
    float radius = center_radius.w;

    vec3 to_center = center - camera_position;
    float t = dot(to_center, ray);

    if (t <= 0.0) {
        return 0.0;
    }

    vec3 sample_position = camera_position + ray * t;
    vec3 q = (sample_position - center) / radius;

    float edge = 1.0 - smoothstep(0.55, 1.0, length(q));
    float clouds = smoothstep(0.32, 0.68, fbm(q * 4.0 + center));
    float wisps = smoothstep(0.42, 0.74, fbm(q * 9.0 + center * 1.7));

    float density = edge * clouds * (0.65 + 0.35 * wisps);
    return density * tint.a;
}

void main() {
    vec2 ndc = (gl_FragCoord.xy / resolution) * 2.0 - 1.0;

    vec4 view_position = inverse_projection * vec4(ndc, 1.0, 1.0);
    vec3 view_ray = normalize(view_position.xyz / view_position.w);
    vec3 world_ray = normalize(
        (inverse_view * vec4(view_ray, 0.0)).xyz
    );

    vec3 color = vec3(0.004, 0.006, 0.018);

    for (int i = 0; i < 32; i++) {
        float density = nebula_blob(
            world_ray,
            nebula_centers[i],
            nebula_colors[i]
        );
        color += nebula_colors[i].rgb * density;
    }

    frag_color = vec4(color, 1.0);
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

system_names = [
    "The Trinary",
    "Neonwake",
    "Glass Meridian",
    "Rustbelt Reach",
    "Blue Static",
    "Ghostline",
    "Chrome Orchard",
    "Nightmarket",
    "The Major Leagues",
]

system_star_indices = [list(range(len(star_masses)))]
system_planet_indices = [list(range(len(planet_masses)))]

system_rng = np.random.default_rng(20261109)
extra_star_colors = []
extra_star_sizes = []
extra_planet_sizes = []

planet_positions = list(planet_positions)
planet_velocities = list(planet_velocities)

base_masses = np.concatenate([star_masses, planet_masses])
base_positions = np.vstack([star_positions, planet_positions])
base_velocities = np.vstack([star_velocities, planet_velocities])

central_mass = float(np.sum(base_masses))
central_position = np.sum(
    base_masses[:, None] * base_positions, axis=0
) / central_mass
central_velocity = np.sum(
    base_masses[:, None] * base_velocities, axis=0
) / central_mass

rocky_colors = [
    (0.62, 0.48, 0.38),
    (0.72, 0.34, 0.22),
    (0.76, 0.68, 0.50),
    (0.48, 0.52, 0.55),
]
gas_colors = [
    (0.78, 0.56, 0.36),
    (0.67, 0.72, 0.78),
    (0.82, 0.72, 0.54),
    (0.57, 0.64, 0.72),
]

extra_star_masses = []
extra_star_densities = []
extra_star_positions = []
extra_star_velocities = []

for system_index, system_name in enumerate(system_names[1:]):
    direction = system_rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    reference = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(direction, reference)
    if np.linalg.norm(tangent) < 1e-8:
        reference = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(direction, reference)
    tangent /= np.linalg.norm(tangent)

    is_major_leagues = system_name == "The Major Leagues"

    distance = (
        400.0
        if is_major_leagues
        else system_rng.uniform(25.0, 200.0)
    )
    system_position = central_position + direction * distance
    system_velocity = (
        central_velocity
        + tangent * np.sqrt(G * central_mass / distance)
    )

    star_mass = (
        2.0
        if is_major_leagues
        else float(np.exp(system_rng.uniform(np.log(0.2), np.log(1.4))))
    )
    companion_mass = 1.6 if is_major_leagues else 0.0
    planet_central_mass = star_mass + companion_mass

    first_star_index = len(star_masses) + len(extra_star_masses)

    if is_major_leagues:
        binary_separation = 0.12
        binary_normal = np.cross(direction, tangent)
        binary_normal /= np.linalg.norm(binary_normal)
        relative_speed = np.sqrt(
            G * planet_central_mass / binary_separation
        )

        primary_position = (
            system_position
            - (companion_mass / planet_central_mass)
            * binary_separation * tangent
        )
        primary_velocity = (
            system_velocity
            - (companion_mass / planet_central_mass)
            * relative_speed * binary_normal
        )
    else:
        primary_position = system_position
        primary_velocity = system_velocity

    extra_star_masses.append(star_mass)
    extra_star_densities.append(1.41 * star_mass ** -1.4)
    extra_star_positions.append(primary_position)
    extra_star_velocities.append(primary_velocity)

    if star_mass < 0.45:
        extra_star_colors.append((1.0, 0.32, 0.12))
    elif star_mass < 0.8:
        extra_star_colors.append((1.0, 0.66, 0.38))
    elif star_mass < 1.1:
        extra_star_colors.append((1.0, 0.91, 0.72))
    else:
        extra_star_colors.append((0.66, 0.78, 1.0))

    extra_star_sizes.append(
        100.0 if is_major_leagues
        else float(np.clip(42.0 * star_mass**0.8, 16, 48))
    )

    if is_major_leagues:
        extra_star_masses.append(companion_mass)
        extra_star_densities.append(1.41 * companion_mass ** -1.4)
        extra_star_positions.append(
            system_position
            + (star_mass / planet_central_mass)
            * binary_separation * tangent
        )
        extra_star_velocities.append(
            system_velocity
            + (star_mass / planet_central_mass)
            * relative_speed * binary_normal
        )
        extra_star_colors.append((1.0, 0.66, 0.38))
        extra_star_sizes.append(90.0)
        system_star_indices.append([first_star_index, first_star_index + 1])
        planet_count = 7
        inner_radius = 0.55
        outer_radius = 4.0
    else:
        system_star_indices.append([first_star_index])
        planet_count = int(system_rng.integers(2, 9))
        inner_radius = system_rng.uniform(0.08, 0.18)
        outer_radius = system_rng.uniform(0.8, 3.0)

    orbit_radii = np.geomspace(inner_radius, outer_radius, planet_count)
    orbit_radii *= system_rng.uniform(0.92, 1.08, size=planet_count)
    orbit_radii.sort()

    this_system_planets = []

    for planet_number, orbit_radius in enumerate(orbit_radii, start=1):
        planet_index = len(planet_masses)
        this_system_planets.append(planet_index)

        is_gas = bool(system_rng.random() < 0.35)
        planet_types.append("gas" if is_gas else "rocky")
        planet_names.append(f"{system_name} {planet_number}")

        if is_gas:
            planet_masses = np.append(
                planet_masses, system_rng.uniform(2e-5, 5e-4)
            )
            planet_densities = np.append(
                planet_densities, system_rng.uniform(0.4, 1.8)
            )
            planet_colors.append(
                gas_colors[int(system_rng.integers(len(gas_colors)))]
            )
            extra_planet_sizes.append(
                float(system_rng.uniform(8.0, 14.0))
            )
        else:
            planet_masses = np.append(
                planet_masses, system_rng.uniform(1e-6, 2e-5)
            )
            planet_densities = np.append(
                planet_densities, system_rng.uniform(3.0, 7.0)
            )
            planet_colors.append(
                rocky_colors[int(system_rng.integers(len(rocky_colors)))]
            )
            extra_planet_sizes.append(
                float(system_rng.uniform(5.0, 9.0))
            )

        planet_orbit_radii = np.append(planet_orbit_radii, orbit_radius)

        angle = system_rng.uniform(0.0, 2.0 * math.pi)
        normal = system_rng.normal(size=3)
        normal /= np.linalg.norm(normal)

        axis_u = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(axis_u) < 1e-8:
            axis_u = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)

        radial = np.cos(angle) * axis_u + np.sin(angle) * axis_v
        orbital_tangent = -np.sin(angle) * axis_u + np.cos(angle) * axis_v

        planet_positions.append(
            system_position + orbit_radius * radial
        )
        planet_velocities.append(
         system_velocity
        + np.sqrt(G * planet_central_mass / orbit_radius) * orbital_tangent
)

    system_planet_indices.append(this_system_planets)

star_masses = np.concatenate([
    star_masses,
    np.asarray(extra_star_masses, dtype=np.float64),
])
star_densities = np.concatenate([
    star_densities,
    np.asarray(extra_star_densities, dtype=np.float64),
])
star_positions = np.vstack([
    star_positions,
    np.asarray(extra_star_positions, dtype=np.float64),
])
star_velocities = np.vstack([
    star_velocities,
    np.asarray(extra_star_velocities, dtype=np.float64),
])

planet_positions = np.asarray(planet_positions, dtype=np.float64)
planet_velocities = np.asarray(planet_velocities, dtype=np.float64)

NUM_STARS = len(star_masses)
NUM_PLANETS = len(planet_masses)
NUM_BODIES = NUM_STARS + NUM_PLANETS
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
] + extra_star_colors

body_colors = star_colors + planet_colors

body_sizes = np.array(
    [42, 38, 48] + extra_star_sizes
    + [5, 5, 6, 6, 6, 7, 9, 10, 11, 12]
    + extra_planet_sizes,
    dtype=np.float32,
)
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
# Planet volumes in km cube
planet_volumes_km3 = (
    (4.0 / 3.0) * math.pi * (planet_radii_au * AU_METERS / 1000.0)**3
)
initial_planet_volumes_km3 = planet_volumes_km3.copy()
#update planet physics
def update_planet_physics(body_index):
    planet_index = body_index - NUM_STARS

    volume_m3 = planet_volumes_km3[planet_index] * 1e9
    mass_kg = masses[body_index] * SOLAR_MASS_KG
    density_kg_m3 = mass_kg / volume_m3
    planet_densities[planet_index] = density_kg_m3 / 1000.0

    radius_km = (
        3.0 * planet_volumes_km3[planet_index] / (4.0 * math.pi)
    ) ** (1.0 / 3.0)

    radius_au = radius_km * 1000.0 / AU_METERS
    planet_radii_au[planet_index] = radius_au
    collision_radii[body_index] = radius_au

    return planet_densities[planet_index]



# ============================================================
# GRAVITY 
# ============================================================

def acceleration(pos):
    # delta[i, j] points from body i toward body j.
    delta = pos[None, :, :] - pos[:, None, :]
    distance = np.linalg.norm(delta, axis=2)

    valid = distance >= 1e-12
    safe_distance = np.maximum(distance, 1e-12)
    source_radii = np.maximum(collision_radii[None, :], 1e-12)

    point_mass_factor = (
        G * masses[None, :] / safe_distance**3
    )
    inside_sphere_factor = (
        G * masses[None, :] / source_radii**3
    )

    factors = np.where(
        distance >= source_radii,
        point_mass_factor,
        inside_sphere_factor,
    )
    factors[~valid] = 0.0

    return np.sum(delta * factors[:, :, None], axis=1)


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

        if body_index >= NUM_STARS:
            update_planet_physics(body_index)

        for i in range(NUM_BODIES):
            history[i].clear()
            history[i].append(
                (simulation_years, positions[i].copy())
            )

    elif property_name == "volume":
        planet_index = body_index - NUM_STARS
        planet_volumes_km3[planet_index] = app_data
        update_planet_physics(body_index)

    elif property_name == "size":
        body_sizes[body_index] = app_data

#reset
def reset_simulation():
    global simulation_years, paused, last_trail_update

    positions[:] = initial_positions
    velocities[:] = initial_velocities
    masses[:] = initial_masses
    body_sizes[:] = initial_body_sizes
    planet_volumes_km3[:] = initial_planet_volumes_km3

    for body_index in range(NUM_STARS, NUM_BODIES):
        update_planet_physics(body_index)

    simulation_years = 0.0
    paused = False

    for i in range(NUM_BODIES):
        history[i].clear()
        start_position = positions[i].copy()
        history[i].append((simulation_years, start_position.copy()))
        history[i].append((simulation_years + DT, start_position.copy()))

        for component in range(3):
            dpg.set_value(
                f"position_{i}_{component}",
                float(positions[i, component])
            )
            dpg.set_value(
                f"velocity_{i}_{component}",
                float(velocities[i, component])
            )

        dpg.set_value(f"mass_{i}", float(masses[i]))
        dpg.set_value(f"size_{i}", float(body_sizes[i]))

        if i >= NUM_STARS:
            planet_index = i - NUM_STARS
            dpg.set_value(
                f"volume_{i}",
                float(planet_volumes_km3[planet_index])
            )
            dpg.set_value(
                f"density_{i}",
                f"Density: {planet_densities[planet_index]:.3f} g/cm³"
            )

    dpg.set_item_label("pause_button", "Pause")

    # Trail buffers are refreshed by the main/render thread on its next update.
    last_trail_update = 0.0    

def add_body_controls(body_index, body_name):
    with dpg.collapsing_header(label=body_name, default_open=False):

        dpg.add_text("Position (AU)")

        for axis, component in zip(["X", "Y", "Z"], range(3)):
            dpg.add_input_float(
                tag=f"position_{body_index}_{component}",
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
                tag=f"velocity_{body_index}_{component}",
                label=f"Velocity {axis}",
                default_value=float(velocities[body_index, component]),
                callback=set_body_value,
                user_data=("velocity", body_index, component),
                width=200
            )

        dpg.add_separator()

        dpg.add_input_float(
            tag=f"mass_{body_index}",
            label="Mass (solar masses)",
            default_value=float(masses[body_index]),
            callback=set_body_value,
            user_data=("mass", body_index, None),
            width=200
        )

        if body_index >= NUM_STARS:
            planet_index = body_index - NUM_STARS

            dpg.add_input_float(
                tag=f"volume_{body_index}",
                label="Volume (km³)",
                default_value=float(planet_volumes_km3[planet_index]),
                callback=set_body_value,
                user_data=("volume", body_index, None),
                width=200
            )

            dpg.add_text(
                f"Density: {planet_densities[planet_index]:.3f} g/cm³",
                tag=f"density_{body_index}"
            )

        dpg.add_input_float(
            tag=f"size_{body_index}",
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
# PROCEDURAL BACKGROUND STAR CHUNKS
# ============================================================

BACKGROUND_CHUNK_SIZE = 200.0
STARS_PER_CHUNK = 75000
BACKGROUND_CHUNK_OFFSETS = [
    (x, y, z)
    for x in (-1, 0, 1)
    for y in (-1, 0, 1)
    for z in (-1, 0, 1)
]
BACKGROUND_VISIBLE_STAR_COUNT = (
    STARS_PER_CHUNK * len(BACKGROUND_CHUNK_OFFSETS)
)

# Kept for the existing deterministic nebula generation below.
rng = np.random.default_rng(42)


def generate_star_chunk(chunk_x, chunk_y, chunk_z):
    chunk_coordinates = (chunk_x, chunk_y, chunk_z)
    seed = (
        chunk_x * 73856093
        ^ chunk_y * 19349663
        ^ chunk_z * 83492791
    ) & 0xFFFFFFFF
    chunk_rng = np.random.default_rng(seed)

    origin = np.array(chunk_coordinates, dtype=np.float64) * BACKGROUND_CHUNK_SIZE

    # Most stars are scattered evenly; a quarter form loose local clusters.
    star_positions = origin + chunk_rng.random(
        (STARS_PER_CHUNK, 3)
    ) * BACKGROUND_CHUNK_SIZE

    cluster_count = STARS_PER_CHUNK // 4
    cluster_centers = origin + chunk_rng.random(
        (16, 3)
    ) * BACKGROUND_CHUNK_SIZE
    cluster_ids = chunk_rng.integers(0, len(cluster_centers), size=cluster_count)

    clustered_positions = (
        cluster_centers[cluster_ids]
        + chunk_rng.normal(
            0.0,
            BACKGROUND_CHUNK_SIZE * 0.045,
            size=(cluster_count, 3),
        )
    )
    clustered_positions = np.clip(
        clustered_positions,
        origin,
        origin + BACKGROUND_CHUNK_SIZE,
    )
    star_positions[:cluster_count] = clustered_positions

    spectral_classes = chunk_rng.choice(
        4,
        size=STARS_PER_CHUNK,
        p=[0.62, 0.25, 0.11, 0.02],
    )
    palette = np.array([
        (0.72, 0.27, 0.12),  # red
        (1.00, 0.55, 0.28),  # orange
        (0.82, 0.88, 1.00),  # white
        (0.52, 0.68, 1.00),  # blue-white
    ], dtype=np.float32)

    colors = palette[spectral_classes]
    brightness = np.clip(
        chunk_rng.lognormal(
            mean=-0.3,
            sigma=0.75,
            size=STARS_PER_CHUNK,
        ),
        0.12,
        1.5,
    ).astype(np.float32)
    colors = np.clip(
        colors * brightness[:, None],
        0.0,
        1.5,
    )

    class_sizes = np.array([0.8, 0.95, 1.15, 1.5], dtype=np.float32)
    sizes = np.clip(
        class_sizes[spectral_classes]
        * chunk_rng.lognormal(
            mean=0.0,
            sigma=0.35,
            size=STARS_PER_CHUNK,
        ),
        0.7,
        3.5,
    ).astype(np.float32)

    return np.column_stack([
        star_positions.astype(np.float32),
        colors,
        sizes,
    ]).astype(np.float32)


def generate_visible_stars(center_chunk):
    chunk_data = []

    for offset_x, offset_y, offset_z in BACKGROUND_CHUNK_OFFSETS:
        chunk_data.append(generate_star_chunk(
            center_chunk[0] + offset_x,
            center_chunk[1] + offset_y,
            center_chunk[2] + offset_z,
        ))

    return np.vstack(chunk_data).astype(np.float32)
#======================================================
#nebula gen
#======================================================
NUM_NEBULAE = 32

nebula_directions = rng.normal(size=(NUM_NEBULAE, 3))
nebula_directions /= np.linalg.norm(
    nebula_directions, axis=1, keepdims=True
)

nebula_distances = rng.uniform(12.0, 85.0, size=NUM_NEBULAE)
nebula_positions = nebula_directions * nebula_distances[:, None]
nebula_radii = rng.uniform(2.0, 5.0, size=NUM_NEBULAE)

nebula_palette = np.array([
    (0.035, 0.012, 0.075),  # violet
    (0.008, 0.045, 0.055),  # teal
    (0.055, 0.014, 0.025),  # rose
    (0.050, 0.032, 0.008),  # amber
], dtype=np.float32)

nebula_colors = nebula_palette[
    rng.integers(0, len(nebula_palette), size=NUM_NEBULAE)
]

nebula_strengths = rng.uniform(
    0.35, 0.8, size=NUM_NEBULAE
).astype(np.float32)

nebula_centers_data = np.column_stack([
    nebula_positions,
    nebula_radii,
]).astype(np.float32)

nebula_colors_data = np.column_stack([
    nebula_colors,
    nebula_strengths,
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
def draw_system_labels(projection, view, width, height):
    ctx.disable(moderngl.DEPTH_TEST)

    for system_index, star_indices in enumerate(system_star_indices):
        indices = np.asarray(star_indices, dtype=np.intp)
        center = np.average(
            positions[indices],
            axis=0,
            weights=masses[indices],
        )

        clip = projection @ view @ np.append(center, 1.0)

        if clip[3] <= 0.0:
            continue

        ndc = clip[:3] / clip[3]

        if (
            abs(ndc[0]) > 1.0
            or abs(ndc[1]) > 1.0
            or not -1.0 <= ndc[2] <= 1.0
        ):
            continue

        texture, (label_width, label_height) = (
            system_label_textures[system_index]
        )

        # Place the label just above the projected system center.
        label_center = np.array([
            ndc[0],
            ndc[1] + (label_height * 0.5 + 12.0) * 2.0 / height,
        ])

        label_size = np.array([
            label_width * 2.0 / width,
            label_height * 2.0 / height,
        ])

        label_program["center_ndc"].value = tuple(label_center)
        label_program["size_ndc"].value = tuple(label_size)
        texture.use(location=0)
        label_vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4)

    ctx.enable(moderngl.DEPTH_TEST)


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
ctx.enable(moderngl.BLEND)
ctx.blend_func = (
    moderngl.SRC_ALPHA,
    moderngl.ONE_MINUS_SRC_ALPHA,
)

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
nebula_program = ctx.program(
    vertex_shader=NEBULA_VERTEX_SHADER,
    fragment_shader=NEBULA_FRAGMENT_SHADER,
)
nebula_program["nebula_centers"].write(
    nebula_centers_data.tobytes()
)
nebula_program["nebula_colors"].write(
    nebula_colors_data.tobytes()
)
nebula_vao = ctx.vertex_array(nebula_program, [])
label_program = ctx.program(
    vertex_shader=LABEL_VERTEX_SHADER,
    fragment_shader=LABEL_FRAGMENT_SHADER,
)
label_program["label_texture"].value = 0
label_vao = ctx.vertex_array(label_program, [])

label_font = ImageFont.truetype(
    r"C:\Windows\Fonts\arialbd.ttf",
    18,
)

system_label_textures = []

for system_name in system_names:
    label_text = system_name.upper()
    bounds = label_font.getbbox(label_text, stroke_width=1)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]

    image_width = text_width + 28
    image_height = text_height + 18

    image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (1, 1, image_width - 2, image_height - 2),
        radius=4,
        fill=(4, 12, 23, 205),
        outline=(45, 205, 220, 220),
        width=1,
    )
    draw.line(
        (7, 3, image_width - 8, 3),
        fill=(255, 166, 70, 235),
        width=2,
    )
    draw.text(
        (14 - bounds[0], 9 - bounds[1]),
        label_text,
        font=label_font,
        fill=(185, 241, 248, 255),
        stroke_width=1,
        stroke_fill=(2, 8, 16, 230),
    )

    texture = ctx.texture(image.size, 4, image.tobytes(), alignment=1)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    system_label_textures.append((texture, image.size))

for system_name in system_names:
    bounds = label_font.getbbox(system_name, stroke_width=2)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]

    image = Image.new(
        "RGBA",
        (text_width + 24, text_height + 20),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(image)
    draw.text(
        (12 - bounds[0], 10 - bounds[1]),
        system_name,
        font=label_font,
        fill=(220, 235, 255, 255),
        stroke_width=2,
        stroke_fill=(3, 8, 20, 240),
    )

    texture = ctx.texture(image.size, 4, image.tobytes(), alignment=1)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    system_label_textures.append((texture, image.size))

print("OpenGL renderer:", ctx.info.get("GL_RENDERER", "Unknown"))
print("Bodies: 3 stars, 6 rocky planets, 4 gas giants")
print("Drag mouse to rotate; scroll to zoom; ESC to exit.")

# Background star GPU buffers.

background_chunks = {}
pending_background_chunks = []
loaded_background_chunk = None


def load_background_chunks(center_chunk):
    global loaded_background_chunk, pending_background_chunks

    if center_chunk != loaded_background_chunk:
        wanted = {
            (
                center_chunk[0] + offset[0],
                center_chunk[1] + offset[1],
                center_chunk[2] + offset[2],
            )
            for offset in BACKGROUND_CHUNK_OFFSETS
        }

        for chunk_key in list(background_chunks):
            if chunk_key not in wanted:
                buffer, vao = background_chunks.pop(chunk_key)
                vao.release()
                buffer.release()

        pending_background_chunks = [
            chunk_key
            for chunk_key in wanted
            if chunk_key not in background_chunks
        ]

        pending_background_chunks.sort(
            key=lambda key: sum(
                (key[axis] - center_chunk[axis]) ** 2
                for axis in range(3)
            )
        )
        loaded_background_chunk = center_chunk

    if pending_background_chunks:
        chunk_key = pending_background_chunks.pop(0)
        data = generate_star_chunk(*chunk_key)
        buffer = ctx.buffer(data.tobytes())
        vao = ctx.vertex_array(
            point_program,
            [(buffer, "3f 3f 1f",
              "in_position", "in_color", "in_size")],
        )
        background_chunks[chunk_key] = (buffer, vao)


def chunk_is_visible(chunk_key, forward, right, camera_up, aspect, far_plane):
    chunk_center = (
        np.asarray(chunk_key, dtype=np.float64) + 0.5
    ) * BACKGROUND_CHUNK_SIZE
    relative = chunk_center - camera_position

    depth = float(np.dot(relative, forward))
    radius = BACKGROUND_CHUNK_SIZE * np.sqrt(3.0) * 0.5

    if depth + radius < 0.1 or depth - radius > far_plane:
        return False

    tan_half_vertical = math.tan(math.radians(60.0) * 0.5)
    half_height = max(depth, 0.0) * tan_half_vertical
    half_width = half_height * aspect

    horizontal = abs(float(np.dot(relative, right)))
    vertical = abs(float(np.dot(relative, camera_up)))

    return (
        horizontal <= half_width + radius
        and vertical <= half_height + radius
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
loaded_background_chunk = None
travel_requests = SimpleQueue()
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
    
def request_travel(sender, app_data, system_index):
    travel_requests.put(system_index)


def travel_to_system(system_index):
    global camera_position, camera_yaw, camera_pitch

    star_indices = np.asarray(
        system_star_indices[system_index],
        dtype=np.intp,
    )
    system_masses = masses[star_indices]
    center = np.average(positions[star_indices], axis=0, weights=system_masses)

    planet_indices = system_planet_indices[system_index]
    largest_orbit = float(np.max(planet_orbit_radii[planet_indices]))
    viewing_distance = max(12.0, largest_orbit * 5.0)

    offset = np.array([0.0, viewing_distance * 0.25, viewing_distance])
    camera_position = center + offset

    direction = center - camera_position
    direction /= np.linalg.norm(direction)
    camera_yaw = math.atan2(direction[2], direction[0])
    camera_pitch = math.asin(float(direction[1]))   


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

    with dpg.theme() as hud_theme:
     with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(
                dpg.mvThemeCol_Text,
                (190, 225, 235, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_WindowBg,
                (7, 12, 22, 250),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ChildBg,
                (10, 18, 30, 245),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Border,
                (30, 165, 185, 180),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg,
                (14, 28, 42, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered,
                (20, 75, 88, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive,
                (22, 115, 125, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Button,
                (12, 55, 70, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered,
                (18, 115, 130, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive,
                (25, 155, 165, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Header,
                (12, 48, 62, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered,
                (18, 95, 110, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive,
                (20, 125, 135, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_CheckMark,
                (45, 225, 225, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrab,
                (35, 185, 200, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrabActive,
                (255, 170, 65, 255),
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowRounding,
                5,
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_FrameRounding,
                3,
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_GrabRounding,
                3,
            )

    dpg.bind_theme(hud_theme)

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
        dpg.add_checkbox(
           label="Show Planets",
            default_value=show_planets,
            callback=lambda s, a: set_show_planets(a)
        )

        dpg.add_slider_float(
            label="Star Size",
            min_value=0.25,
            max_value=4.0,
            default_value=star_scale,
            callback=lambda s, a: set_star_scale(a)
        )

        dpg.add_slider_float(
            label="Planet Size",
            min_value=0.25,
            max_value=4.0,
            default_value=planet_scale,
            callback=lambda s, a: set_planet_scale(a)
        )

        dpg.add_separator()

        dpg.add_text("STAR SYSTEMS")

        for system_index, system_name in enumerate(system_names):
            with dpg.collapsing_header(
                label=system_name,
                default_open=(system_index == 0),
            ):
                dpg.add_button(
                    label="Travel to system",
                    callback=request_travel,
                    user_data=system_index,
                )

                for star_index in system_star_indices[system_index]:
                    add_body_controls(star_index, f"{system_name} Star")

                for planet_index in system_planet_indices[system_index]:
                    body_index = NUM_STARS + planet_index
                    add_body_controls(
                        body_index,
                        planet_names[planet_index],
                    )
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
        while not travel_requests.empty():
            travel_to_system(travel_requests.get())

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

        for i in range(NUM_BODIES):
            displayed_positions[i] = apparent_position(
                i, simulation_years, history
            )

        displayed_sizes = body_sizes.copy()
        displayed_sizes[:NUM_STARS] *= star_scale
        displayed_sizes[NUM_STARS:] *= planet_scale

        body_data = np.hstack([
            displayed_positions.astype(np.float32),
            np.array(body_colors, dtype=np.float32),
            displayed_sizes[:, None],
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

        far_plane = max(
            200.0,
            float(np.max(np.linalg.norm(
                positions - camera_position,
                axis=1,
            ))) + 100.0,
        )
        near_plane = max(0.01, far_plane * 1e-5)

        projection = perspective(
            60.0,
            width / height,
            near_plane,
            far_plane,
        )

        
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
        nebula_program["resolution"].value = (float(width), float(height))
        nebula_program["inverse_projection"].write(
            np.linalg.inv(projection).T.astype("f4").tobytes()
        )
        nebula_program["inverse_view"].write(
            np.linalg.inv(view).T.astype("f4").tobytes()
        )
        nebula_program["camera_position"].value = tuple(
            float(value) for value in camera_position
        )
        nebula_vao.render(mode=moderngl.TRIANGLES, vertices=3)

            # Background stars
        current_chunk = tuple(
            np.floor(
                camera_position / BACKGROUND_CHUNK_SIZE
            ).astype(np.int64)
        )

        load_background_chunks(current_chunk)

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)

        for chunk_key, (buffer, vao) in background_chunks.items():
            if chunk_is_visible(
                chunk_key,
                forward,
                right,
                camera_up,
                width / height,
                far_plane,
            ):
                vao.render(
                    mode=moderngl.POINTS,
                    vertices=STARS_PER_CHUNK,
                )

        # Trajectory trails
        if show_trails:
            for i, vao in enumerate(trail_vaos):
                if vao is not None and (show_planets or i < NUM_STARS):
                    vao.render(mode=moderngl.LINE_STRIP)

        # Main bodies
        body_vao.render(
            mode=moderngl.POINTS,
            vertices=NUM_BODIES if show_planets else NUM_STARS,
        )
        draw_system_labels(projection, view, width, height)

        glfw.swap_buffers(window)

        glfw.set_window_title(
            window,
            f"Three Stars + Ten Planets | {simulation_years:.5f} years"
        )   

finally:
    for texture, _ in system_label_textures:
        texture.release()

    label_vao.release()
    label_program.release()
    for vao in trail_vaos:
        if vao is not None:
            vao.release()

    for buf in trail_buffers:
        if buf is not None:
            buf.release()

    for buffer, vao in background_chunks.values():
     vao.release()
     buffer.release()

    body_vao.release()
    body_buffer.release()
    point_program.release()
    line_program.release()
    body_program.release()
    ctx.release()


    glfw.destroy_window(window)
    glfw.terminate()
