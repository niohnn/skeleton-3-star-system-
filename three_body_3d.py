import math
import os
import sys
import time
from pathlib import Path

import dearpygui.dearpygui as dpg
import glfw
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# SETTINGS
# ============================================================

WIDTH, HEIGHT = 1400, 900
DT = 0.00001
STEPS_PER_FRAME = 10
TRAIL_LENGTH = 1800
HISTORY_TRIM_BATCH = 128
TRAIL_UPDATE_INTERVAL = 0.025

G = 4.0 * math.pi**2  # AU^3 / (solar mass * year^2)
C_AU_PER_YEAR = 63241.077
DAYS_PER_YEAR = 365.25
SOLAR_MASS_KG = 1.98847e30
AU_METERS = 1.495978707e11
FOV_DEGREES = 60.0
WORLD_UP = np.array([0.0, 1.0, 0.0])
MAX_STAR_LIGHTS = 32
NUM_NEBULAE = 32

# Simulation controls
paused = False
time_scale = 1.0
star_scale = 1.0
planet_scale = 1.0
show_planets = True
show_trails = True
control_panel_open = False

# Control panel palette (RGBA). Every fill behind text keeps >= 4.5:1 contrast with HUD_TEXT.
HUD_TEXT = (190, 225, 235, 255)
HUD_ACCENT = (35, 185, 200, 255)
HUD_ACCENT_ACTIVE = (255, 170, 65, 255)
HUD_COLORS = {
    "Text": HUD_TEXT,
    "WindowBg": (7, 12, 22, 250),
    "ChildBg": (10, 18, 30, 245),
    "Border": (30, 165, 185, 180),
    "FrameBg": (14, 28, 42, 255),
    "FrameBgHovered": (20, 75, 88, 255),
    "FrameBgActive": (20, 105, 114, 255),
    "Button": (12, 55, 70, 255),
    "ButtonHovered": (15, 88, 102, 255),
    "ButtonActive": (16, 105, 112, 255),
    "Header": (12, 48, 62, 255),
    "HeaderHovered": (16, 80, 94, 255),
    "HeaderActive": (16, 104, 113, 255),
    "CheckMark": (45, 225, 225, 255),
    "SliderGrab": HUD_ACCENT,
    "SliderGrabActive": HUD_ACCENT_ACTIVE,
}
PANEL_FONT_SIZE = 16
PANEL_INPUT_WIDTH = -200  # negative: fill the row, leave this many pixels for the label
PANEL_FONT_PATHS = (
    r"C:\Windows\Fonts\arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
ENERGY_ERROR_GOOD = 1e-6
ENERGY_ERROR_POOR = 1e-3
ESC_CONFIRM_SECONDS = 1.5
CAMERA_SPEED = 2.0  # AU per second
CAMERA_BOOST = 10.0  # speed multiplier while Shift is held
MOUSE_SENSITIVITY = 0.003

# ============================================================
# SHADERS
# ============================================================

SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def load_shader(name, defines=None, include=None):
    """Read shaders/<name>; #defines and an included snippet go right after #version."""
    version, _, body = (SHADER_DIR / name).read_text().partition("\n")
    header = "".join(f"#define {key} {value}\n" for key, value in (defines or {}).items())
    if include is not None:
        header += (SHADER_DIR / include).read_text()
    return f"{version}\n{header}{body}"


LABEL_VERTEX_SHADER = load_shader("label.vert")
LABEL_FRAGMENT_SHADER = load_shader("label.frag")
POINT_VERTEX_SHADER = load_shader("point.vert")
POINT_FRAGMENT_SHADER = load_shader("point.frag")
LINE_VERTEX_SHADER = load_shader("line.vert")
LINE_FRAGMENT_SHADER = load_shader("line.frag")
BODY_VERTEX_SHADER = load_shader("body.vert", include="body_noise.glsl")
BODY_FRAGMENT_SHADER = load_shader(
    "body.frag", defines={"MAX_STAR_LIGHTS": MAX_STAR_LIGHTS}, include="body_noise.glsl"
)
NEBULA_VERTEX_SHADER = load_shader("nebula.vert")
NEBULA_FRAGMENT_SHADER = load_shader("nebula.frag", defines={"NUM_NEBULAE": NUM_NEBULAE})

# ============================================================
# CENTRAL TRINARY
# Units: solar masses, g/cm^3, AU, AU/yr. Planets start on circular orbits
# around star 0; the other two stars perturb them.
# ============================================================

# mass, density, position, velocity, colour, reference pixel size
TRINARY_STAR_TABLE = [
    (1.0, 1.41, (-0.8, 0.0, 0.3), (-1.5, -2.5, 1.2), (1.0, 0.82, 0.48), 42.0),
    (0.8, 1.41, (0.8, -0.3, -0.2), (2.5, 1.0, -1.5), (0.55, 0.72, 1.0), 38.0),
    (1.2, 1.41, (0.1, 1.0, 0.7), (-1.0, 0.8, 1.0), (1.0, 0.45, 0.28), 48.0),
]

# name, type, rotation period (h), mass, density, orbit radius, colour, reference pixel size
TRINARY_PLANET_TABLE = [
    ("Rock 1", "rocky", 24.0, 1.0e-6, 5.5, 0.12, (0.75, 0.67, 0.57), 5.0),
    ("Rock 2", "rocky", 36.0, 2.0e-6, 5.2, 0.18, (0.75, 0.35, 0.20), 5.0),
    ("Rock 3", "rocky", 23.93, 3.0e-6, 4.5, 0.25, (0.80, 0.72, 0.48), 6.0),
    ("Rock 4", "rocky", 24.62, 2.5e-6, 5.0, 0.34, (0.55, 0.52, 0.48), 6.0),
    ("Rock 5", "rocky", 18.0, 4.0e-6, 3.9, 0.44, (0.65, 0.38, 0.28), 6.0),
    ("Rock 6", "rocky", 30.0, 5.0e-6, 4.0, 0.56, (0.68, 0.58, 0.42), 7.0),
    ("Gas 1", "gas", 9.93, 1.0e-4, 1.3, 0.72, (0.85, 0.63, 0.40), 9.0),
    ("Gas 2", "gas", 10.7, 2.0e-4, 0.8, 0.88, (0.75, 0.67, 0.50), 10.0),
    ("Gas 3", "gas", 9.9, 3.0e-4, 0.5, 1.05, (0.70, 0.78, 0.85), 11.0),
    ("Gas 4", "gas", 16.1, 5.0e-4, 0.3, 1.22, (0.85, 0.72, 0.58), 12.0),
]

GOLDEN_ANGLE = 2.399963229728653


def circular_planet_orbits(radii, central_mass, inclination_step_deg=2.0):
    """Circular orbits around the origin, spread by the golden angle and slightly inclined."""
    pos, vel = [], []
    for i, radius in enumerate(radii):
        angle = i * GOLDEN_ANGLE
        inclination = math.radians((i - 4.5) * inclination_step_deg)
        radial = np.array(
            [
                math.cos(angle),
                math.sin(angle) * math.cos(inclination),
                math.sin(angle) * math.sin(inclination),
            ]
        )
        tangent = np.array(
            [
                -math.sin(angle),
                math.cos(angle) * math.cos(inclination),
                math.cos(angle) * math.sin(inclination),
            ]
        )
        pos.append(radius * radial)
        vel.append(math.sqrt(G * central_mass / radius) * tangent)
    return np.array(pos), np.array(vel)


def main_sequence_color(mass):
    if mass < 0.45:
        return (1.0, 0.32, 0.12)
    if mass < 0.8:
        return (1.0, 0.66, 0.38)
    if mass < 1.1:
        return (1.0, 0.91, 0.72)
    return (0.66, 0.78, 1.0)


star_masses, star_densities, star_positions, star_velocities, star_colors, star_sizes = map(
    list, zip(*TRINARY_STAR_TABLE, strict=True)
)
(
    planet_names,
    planet_types,
    planet_rotation_period_hours,
    planet_masses,
    planet_densities,
    planet_orbit_radii,
    planet_colors,
    planet_sizes,
) = map(list, zip(*TRINARY_PLANET_TABLE, strict=True))

trinary_offsets, trinary_orbit_velocities = circular_planet_orbits(
    planet_orbit_radii, star_masses[0], inclination_step_deg=5.0
)
planet_positions = list(np.asarray(star_positions[0]) + trinary_offsets)
planet_velocities = list(np.asarray(star_velocities[0]) + trinary_orbit_velocities)

# ============================================================
# OUTER STAR SYSTEMS
# Random single stars with planets on circular orbits around the trinary's
# centre of mass, plus one distant binary ("The Major Leagues").
# ============================================================

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

base_masses = np.array(star_masses + planet_masses)
base_positions = np.vstack([star_positions, planet_positions])
base_velocities = np.vstack([star_velocities, planet_velocities])
central_mass = float(np.sum(base_masses))
central_position = np.sum(base_masses[:, None] * base_positions, axis=0) / central_mass
central_velocity = np.sum(base_masses[:, None] * base_velocities, axis=0) / central_mass

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

system_rng = np.random.default_rng(20261109)

for system_name in system_names[1:]:
    direction = system_rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    reference = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(direction, reference)
    if np.linalg.norm(tangent) < 1e-8:
        reference = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(direction, reference)
    tangent /= np.linalg.norm(tangent)

    is_major_leagues = system_name == "The Major Leagues"

    distance = 400.0 if is_major_leagues else system_rng.uniform(25.0, 200.0)
    system_position = central_position + direction * distance
    system_velocity = central_velocity + tangent * np.sqrt(G * central_mass / distance)

    star_mass = (
        2.0 if is_major_leagues else float(np.exp(system_rng.uniform(np.log(0.2), np.log(1.4))))
    )
    companion_mass = 1.6 if is_major_leagues else 0.0
    planet_central_mass = star_mass + companion_mass

    first_star_index = len(star_masses)

    if is_major_leagues:
        binary_separation = 0.12
        binary_normal = np.cross(direction, tangent)
        binary_normal /= np.linalg.norm(binary_normal)
        relative_speed = np.sqrt(G * planet_central_mass / binary_separation)

        primary_position = (
            system_position - (companion_mass / planet_central_mass) * binary_separation * tangent
        )
        primary_velocity = (
            system_velocity
            - (companion_mass / planet_central_mass) * relative_speed * binary_normal
        )
    else:
        primary_position = system_position
        primary_velocity = system_velocity

    star_masses.append(star_mass)
    star_densities.append(1.41 * star_mass**-1.4)
    star_positions.append(primary_position)
    star_velocities.append(primary_velocity)
    star_colors.append(main_sequence_color(star_mass))
    star_sizes.append(100.0 if is_major_leagues else float(np.clip(42.0 * star_mass**0.8, 16, 48)))

    if is_major_leagues:
        star_masses.append(companion_mass)
        star_densities.append(1.41 * companion_mass**-1.4)
        star_positions.append(
            system_position + (star_mass / planet_central_mass) * binary_separation * tangent
        )
        star_velocities.append(
            system_velocity + (star_mass / planet_central_mass) * relative_speed * binary_normal
        )
        star_colors.append((1.0, 0.66, 0.38))
        star_sizes.append(90.0)
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
        this_system_planets.append(len(planet_masses))

        is_gas = bool(system_rng.random() < 0.35)
        planet_types.append("gas" if is_gas else "rocky")
        planet_names.append(f"{system_name} {planet_number}")

        if is_gas:
            planet_masses.append(system_rng.uniform(2e-5, 5e-4))
            planet_densities.append(system_rng.uniform(0.4, 1.8))
            planet_colors.append(gas_colors[int(system_rng.integers(len(gas_colors)))])
            planet_sizes.append(float(system_rng.uniform(8.0, 14.0)))
            planet_rotation_period_hours.append(10.5)
        else:
            planet_masses.append(system_rng.uniform(1e-6, 2e-5))
            planet_densities.append(system_rng.uniform(3.0, 7.0))
            planet_colors.append(rocky_colors[int(system_rng.integers(len(rocky_colors)))])
            planet_sizes.append(float(system_rng.uniform(5.0, 9.0)))
            planet_rotation_period_hours.append(24.0)

        planet_orbit_radii.append(orbit_radius)

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

        planet_positions.append(system_position + orbit_radius * radial)
        planet_velocities.append(
            system_velocity + np.sqrt(G * planet_central_mass / orbit_radius) * orbital_tangent
        )

    system_planet_indices.append(this_system_planets)

planet_orbit_radii = np.asarray(planet_orbit_radii, dtype=np.float64)

NUM_STARS = len(star_masses)
NUM_PLANETS = len(planet_masses)
NUM_BODIES = NUM_STARS + NUM_PLANETS
if NUM_STARS > MAX_STAR_LIGHTS:
    raise ValueError(f"At most {MAX_STAR_LIGHTS} stars can be used as shader light sources.")

# Stars first, then planets, in one gravitational system with zero total momentum.
masses = np.asarray(star_masses + planet_masses, dtype=np.float64)
densities = np.asarray(star_densities + planet_densities, dtype=np.float64)
positions = np.vstack([star_positions, planet_positions]).astype(np.float64)
velocities = np.vstack([star_velocities, planet_velocities]).astype(np.float64)
velocities -= np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)

star_names = [""] * NUM_STARS
for system_name, star_indices in zip(system_names, system_star_indices, strict=True):
    for k, star_index in enumerate(star_indices):
        suffix = f" {chr(ord('A') + k)}" if len(star_indices) > 1 else ""
        star_names[star_index] = f"{system_name} Star{suffix}"
body_names = star_names + planet_names

# ============================================================
# SPIN
# ============================================================

rotation_rng = np.random.default_rng(20261119)
star_rotation_period_years = np.clip(25.0 / np.sqrt(masses[:NUM_STARS]), 5.0, 60.0) / DAYS_PER_YEAR
planet_rotation_period_years = np.asarray(planet_rotation_period_hours) / (24.0 * DAYS_PER_YEAR)
rotation_period_years = np.concatenate([star_rotation_period_years, planet_rotation_period_years])
rotation_phases = rotation_rng.uniform(0.0, 2.0 * math.pi, NUM_BODIES)
rotation_axes = np.zeros((NUM_BODIES, 3), dtype=np.float64)

for star_index in range(NUM_STARS):
    axis = rotation_rng.normal(size=3)
    rotation_axes[star_index] = axis / np.linalg.norm(axis)

# Planet spin axes: tilted up to 30 degrees from the orbit normal.
for system_index, planet_indices in enumerate(system_planet_indices):
    star_indices = np.asarray(system_star_indices[system_index], dtype=np.intp)
    system_center = np.average(positions[star_indices], axis=0, weights=masses[star_indices])
    system_velocity = np.average(velocities[star_indices], axis=0, weights=masses[star_indices])

    for planet_index in planet_indices:
        body_index = NUM_STARS + planet_index
        relative_position = positions[body_index] - system_center
        relative_velocity = velocities[body_index] - system_velocity
        orbit_normal = np.cross(relative_position, relative_velocity)
        orbit_normal /= np.linalg.norm(orbit_normal)

        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(orbit_normal, reference))) > 0.9:
            reference = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(orbit_normal, reference)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(orbit_normal, tangent)

        tilt = rotation_rng.uniform(0.0, math.radians(30.0))
        azimuth = rotation_rng.uniform(0.0, 2.0 * math.pi)
        tilt_direction = math.cos(azimuth) * tangent + math.sin(azimuth) * bitangent
        rotation_axes[body_index] = math.cos(tilt) * orbit_normal + math.sin(tilt) * tilt_direction

# ============================================================
# PHYSICAL RADII AND DISPLAY SETTINGS
# Radii follow from mass and density. Display exaggeration keeps the spheres
# visible at astronomical scales; physics always uses collision_radii.
# ============================================================

MIN_MASS_SOLAR = 1e-12
MIN_VOLUME_KM3 = 1e-12
MIN_VISUAL_SCALE = 0.01
MIN_ROTATION_PERIOD_DAYS = 0.01


def sphere_radius_au(mass, density):
    """Radius of a uniform sphere; mass in solar masses, density in g/cm^3."""
    mass_kg = mass * SOLAR_MASS_KG
    density_kg_m3 = density * 1000.0
    return (3.0 * mass_kg / (4.0 * math.pi * density_kg_m3)) ** (1.0 / 3.0) / AU_METERS


collision_radii = sphere_radius_au(masses, densities)
volumes_km3 = (4.0 / 3.0) * math.pi * (collision_radii * AU_METERS / 1000.0) ** 3


def update_body_physics(body_index=slice(None)):
    """Recompute density and radius from mass and volume (one body, or all by default)."""
    volume_m3 = volumes_km3[body_index] * 1e9
    densities[body_index] = masses[body_index] * SOLAR_MASS_KG / volume_m3 / 1000.0
    radius_km = (3.0 * volumes_km3[body_index] / (4.0 * math.pi)) ** (1.0 / 3.0)
    collision_radii[body_index] = radius_km * 1000.0 / AU_METERS


body_colors = star_colors + planet_colors
body_reference_pixel_sizes = np.asarray(star_sizes + planet_sizes, dtype=np.float32)
REFERENCE_CAMERA_DISTANCE_AU = 4.5
REFERENCE_PROJECTION_SCALE = 1.0 / math.tan(math.radians(FOV_DEGREES) * 0.5)
body_visual_exaggeration = (
    body_reference_pixel_sizes
    * REFERENCE_CAMERA_DISTANCE_AU
    / (REFERENCE_PROJECTION_SCALE * HEIGHT * collision_radii)
)
body_visual_scales = np.ones(NUM_BODIES, dtype=np.float32)
body_is_star = np.zeros(NUM_BODIES, dtype=np.float32)
body_is_star[:NUM_STARS] = 1.0
body_is_gas = np.zeros(NUM_BODIES, dtype=np.float32)
body_is_gas[NUM_STARS:] = [planet_type == "gas" for planet_type in planet_types]

shape_rng = np.random.default_rng(777)
body_seeds = shape_rng.uniform(0.0, 100.0, NUM_BODIES).astype(np.float32)
body_oblateness = np.zeros(NUM_BODIES, dtype=np.float32)
body_roughness = np.zeros(NUM_BODIES, dtype=np.float32)

for planet_index, planet_type in enumerate(planet_types):
    body_index = NUM_STARS + planet_index
    if planet_type == "gas":
        # Faster spin -> flatter planet.
        period_hours = planet_rotation_period_hours[planet_index]
        body_oblateness[body_index] = float(np.clip(1.0 / period_hours, 0.03, 0.12))
    else:
        body_roughness[body_index] = shape_rng.uniform(0.03, 0.10)

initial_positions = positions.copy()
initial_velocities = velocities.copy()
initial_masses = masses.copy()
initial_volumes_km3 = volumes_km3.copy()
initial_body_visual_scales = body_visual_scales.copy()
initial_rotation_period_years = rotation_period_years.copy()
initial_rotation_phases = rotation_phases.copy()


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

    point_mass_factor = G * masses[None, :] / safe_distance**3
    inside_sphere_factor = G * masses[None, :] / source_radii**3

    factors = np.where(
        distance >= source_radii,
        point_mass_factor,
        inside_sphere_factor,
    )
    factors[~valid] = 0.0

    return np.sum(delta * factors[:, :, None], axis=1)


cached_acceleration = None


def invalidate_physics_cache():
    """Call after any edit to positions, masses or radii."""
    global cached_acceleration, diagnostics_rebaseline
    cached_acceleration = None
    diagnostics_rebaseline = True


# ============================================================
# INTEGRATORS
# Each advances pos/vel in place by dt. a_start is the acceleration at the
# incoming positions; the return value is the acceleration at the outgoing
# positions, or None when the method does not produce it.
# ============================================================


def step_euler(pos, vel, dt, a_start):
    # Explicit Euler: 1st order, energy grows without bound.
    pos += vel * dt
    vel += a_start * dt
    return None


def step_symplectic_euler(pos, vel, dt, a_start):
    # Kick then drift: 1st order but symplectic, so energy error stays bounded.
    vel += a_start * dt
    pos += vel * dt
    return None


def step_verlet(pos, vel, dt, a_start):
    # Velocity Verlet (kick-drift-kick): 2nd order, symplectic, time-reversible.
    vel += 0.5 * a_start * dt
    pos += vel * dt
    a_end = acceleration(pos)
    vel += 0.5 * a_end * dt
    return a_end


def step_rk4(pos, vel, dt, a_start):
    # Classic Runge-Kutta: 4th order, but not symplectic -> slow secular energy drift.
    x0, v0 = pos.copy(), vel.copy()
    k2x, k2v = v0 + 0.5 * dt * a_start, acceleration(x0 + 0.5 * dt * v0)
    k3x, k3v = v0 + 0.5 * dt * k2v, acceleration(x0 + 0.5 * dt * k2x)
    k4x, k4v = v0 + dt * k3v, acceleration(x0 + dt * k3x)
    pos[:] = x0 + dt / 6.0 * (v0 + 2.0 * k2x + 2.0 * k3x + k4x)
    vel[:] = v0 + dt / 6.0 * (a_start + 2.0 * k2v + 2.0 * k3v + k4v)
    return None


YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_W0 = 1.0 - 2.0 * YOSHIDA_W1


def step_yoshida4(pos, vel, dt, a_start):
    # Yoshida (1990): three Verlet sub-steps (one backwards) give 4th order, still symplectic.
    a = a_start
    for weight in (YOSHIDA_W1, YOSHIDA_W0, YOSHIDA_W1):
        a = step_verlet(pos, vel, weight * dt, a)
    return a


INTEGRATORS = {
    "Euler (1st order)": step_euler,
    "Symplectic Euler (1st order)": step_symplectic_euler,
    "Velocity Verlet (2nd order)": step_verlet,
    "RK4 (4th order, non-symplectic)": step_rk4,
    "Yoshida (4th order, symplectic)": step_yoshida4,
}
integrator_name = "Velocity Verlet (2nd order)"
TIMESTEP_OPTIONS = ["1e-06", "3e-06", "1e-05", "3e-05", "1e-04"]


def set_integrator(value):
    global integrator_name
    integrator_name = value
    invalidate_physics_cache()


def set_timestep(value):
    global DT
    DT = float(value)
    invalidate_physics_cache()


def physics_step():
    """Advance one DT; on a collision, undo the step, pause and return False."""
    global cached_acceleration

    previous_positions = positions.copy()
    previous_velocities = velocities.copy()

    a0 = cached_acceleration if cached_acceleration is not None else acceleration(positions)
    a_end = INTEGRATORS[integrator_name](positions, velocities, DT, a0)

    collision = detect_collision(positions, previous_positions)
    if collision is not None:
        positions[:] = previous_positions
        velocities[:] = previous_velocities
        cached_acceleration = a0
        body_a, body_b = (body_names[index] for index in collision)
        set_paused(True)
        show_status(f"Collision: {body_a} and {body_b}. Paused before overlap.")
        return False

    cached_acceleration = a_end
    return True


# ============================================================
# CONSERVATION DIAGNOSTICS
# A symplectic integrator keeps the energy error bounded; linear and
# angular momentum should stay constant to round-off.
# ============================================================

DIAGNOSTICS_INTERVAL = 0.25  # wall-clock seconds between panel updates
DIAGNOSTICS_HISTORY = 600
diagnostics_rebaseline = True
diagnostics_baseline = None
last_diagnostics_update = 0.0
energy_error_times = []
energy_error_values = []


def conserved_quantities():
    """Return total energy (M_sun AU^2/yr^2), linear and angular momentum vectors."""
    kinetic = 0.5 * float(np.sum(masses * np.einsum("ij,ij->i", velocities, velocities)))

    i, j = np.triu_indices(NUM_BODIES, k=1)
    distance = np.linalg.norm(positions[i] - positions[j], axis=1)
    potential = -G * float(np.sum(masses[i] * masses[j] / np.maximum(distance, 1e-12)))

    momentum = np.sum(masses[:, None] * velocities, axis=0)
    angular_momentum = np.sum(masses[:, None] * np.cross(positions, velocities), axis=0)
    return kinetic + potential, momentum, angular_momentum


def update_diagnostics(force=False):
    """Recompute conserved quantities and push them to the control panel."""
    global diagnostics_rebaseline, diagnostics_baseline, last_diagnostics_update

    now = time.perf_counter()
    if (
        not force
        and not diagnostics_rebaseline
        and now - last_diagnostics_update < DIAGNOSTICS_INTERVAL
    ):
        return
    last_diagnostics_update = now

    energy, momentum, angular_momentum = conserved_quantities()
    if diagnostics_rebaseline or diagnostics_baseline is None:
        diagnostics_baseline = (energy, momentum.copy(), angular_momentum.copy())
        diagnostics_rebaseline = False
        energy_error_times.clear()
        energy_error_values.clear()

    energy0, _, angular_momentum0 = diagnostics_baseline
    energy_error = abs((energy - energy0) / energy0) if energy0 != 0.0 else 0.0
    angular_error = float(
        np.linalg.norm(angular_momentum - angular_momentum0)
        / max(np.linalg.norm(angular_momentum0), 1e-300)
    )

    # While paused the time stands still; repeated points would only pile up.
    if not energy_error_times or simulation_years != energy_error_times[-1]:
        energy_error_times.append(simulation_years)
        energy_error_values.append(math.log10(max(energy_error, 1e-16)))
    if len(energy_error_times) > DIAGNOSTICS_HISTORY:
        del energy_error_times[0]
        del energy_error_values[0]

    if not control_panel_open:
        return
    dpg.set_value("diag_time", f"t = {simulation_years:.5f} yr   dt = {DT:g} yr")
    dpg.set_value("diag_energy", f"E = {energy:.9e}")
    if energy_error < ENERGY_ERROR_GOOD:
        energy_verdict = "good"
    elif energy_error < ENERGY_ERROR_POOR:
        energy_verdict = "acceptable"
    else:
        energy_verdict = "poor, reduce dt"
    dpg.set_value(
        "diag_energy_error", f"Energy drift |dE/E0| = {energy_error:.3e}  ({energy_verdict})"
    )
    dpg.set_value(
        "diag_momentum", f"Total momentum |P| = {np.linalg.norm(momentum):.3e}  (ideal 0)"
    )
    dpg.set_value("diag_angular_error", f"Angular momentum drift |dL|/|L0| = {angular_error:.3e}")
    dpg.set_value("energy_series", [energy_error_times, energy_error_values])
    dpg.fit_axis_data("energy_x_axis")
    dpg.fit_axis_data("energy_y_axis")


def detect_collision(pos, previous_pos):
    """First pair (i, j) whose spheres touch while moving linearly from previous_pos to pos."""
    pairwise_offsets = previous_pos[:, None, :] - previous_pos[None, :, :]
    minimum_distances = collision_radii[:, None] + collision_radii[None, :]

    # Cheap rejection: the gap can shrink by at most both bodies' step lengths.
    start_distances = np.sqrt(np.einsum("ijk,ijk->ij", pairwise_offsets, pairwise_offsets))
    step_lengths = np.linalg.norm(pos - previous_pos, axis=1)
    gaps = start_distances - step_lengths[:, None] - step_lengths[None, :] - minimum_distances
    np.fill_diagonal(gaps, np.inf)
    if gaps.min() > 1e-9:
        return None

    pairwise_motion = (pos - previous_pos)[:, None, :] - (pos - previous_pos)[None, :, :]
    motion_squared = np.einsum("ijk,ijk->ij", pairwise_motion, pairwise_motion)
    start_motion_dot = np.einsum("ijk,ijk->ij", pairwise_offsets, pairwise_motion)
    closest_time = np.zeros_like(motion_squared)
    np.divide(-start_motion_dot, motion_squared, out=closest_time, where=motion_squared > 1e-20)
    np.clip(closest_time, 0.0, 1.0, out=closest_time)
    closest_offsets = pairwise_offsets + closest_time[:, :, None] * pairwise_motion
    pairwise_distances = np.linalg.norm(closest_offsets, axis=2)

    colliding_pairs = np.argwhere(np.triu(pairwise_distances <= minimum_distances, k=1))
    if len(colliding_pairs) == 0:
        return None
    return tuple(int(index) for index in colliding_pairs[0])


# ============================================================
# SCENARIOS
# Classic three-body initial conditions for the central trinary. Values from
# the literature use G = 1; in G = 4 pi^2 units the same positions and masses
# need velocities scaled by 2 pi (and times divided by 2 pi).
# The trinary's planets are moved onto circular orbits around the stars'
# centre of mass, far enough out to act as near-test particles.
# ============================================================

TRINARY_STARS = list(system_star_indices[0])
TRINARY_PLANETS = list(system_planet_indices[0])
TRINARY_BODIES = TRINARY_STARS + [NUM_STARS + p for p in TRINARY_PLANETS]
SQRT_G = 2.0 * math.pi


def figure_eight_stars():
    # Chenciner & Montgomery (2000); period 6.32591398 (G = 1) = 1.00680 yr here.
    x1 = np.array([0.97000436, -0.24308753, 0.0])
    v3 = SQRT_G * np.array([-0.93240737, -0.86473146, 0.0])
    return np.array([x1, -x1, np.zeros(3)]), np.array([-v3 / 2, -v3 / 2, v3]), np.ones(3)


def lagrange_triangle_stars(side=1.5):
    # Equilateral rigid rotation, omega^2 = G M_total / side^3. Unstable for equal masses.
    masses_ = np.ones(3)
    radius = side / math.sqrt(3.0)
    omega = math.sqrt(G * masses_.sum() / side**3)
    angles = np.radians([90.0, 210.0, 330.0])
    pos = radius * np.stack([np.cos(angles), np.sin(angles), np.zeros(3)], axis=1)
    vel = omega * radius * np.stack([-np.sin(angles), np.cos(angles), np.zeros(3)], axis=1)
    return pos, vel, masses_


def hierarchical_triple_stars(inner=0.4, outer=5.0, inclination_deg=20.0):
    # Tight 1.0 + 0.8 binary orbited by a 1.2 star; stable (outer/inner ratio 12.5).
    m = np.array([1.0, 0.8, 1.2])
    m_in, m_tot = m[0] + m[1], m.sum()
    x_hat = np.array([1.0, 0.0, 0.0])
    y_hat = np.array([0.0, 1.0, 0.0])
    tilt = math.radians(inclination_deg)
    outer_dir = np.array([0.0, math.cos(tilt), math.sin(tilt)])

    v_in = math.sqrt(G * m_in / inner)
    v_out = math.sqrt(G * m_tot / outer)
    binary_pos = -(m[2] / m_tot) * outer * x_hat
    binary_vel = -(m[2] / m_tot) * v_out * outer_dir

    pos = np.array(
        [
            binary_pos - (m[1] / m_in) * inner * x_hat,
            binary_pos + (m[0] / m_in) * inner * x_hat,
            (m_in / m_tot) * outer * x_hat,
        ]
    )
    vel = np.array(
        [
            binary_vel - (m[1] / m_in) * v_in * y_hat,
            binary_vel + (m[0] / m_in) * v_in * y_hat,
            (m_in / m_tot) * v_out * outer_dir,
        ]
    )
    return pos, vel, m


def pythagorean_stars():
    # Burrau (1913): masses 3:4:5 at rest on a 3-4-5 triangle (scaled to 0.6:0.8:1.0).
    # Very close encounters: a fixed timestep fails here (watch |dE/E0|).
    pos = np.array([[1.0, 3.0, 0.0], [-2.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    return pos, np.zeros((3, 3)), np.array([0.6, 0.8, 1.0])


SCENARIO_BUILDERS = {
    "Figure-8 (periodic)": (figure_eight_stars, (4.0, 12.0)),
    "Lagrange triangle (unstable)": (lagrange_triangle_stars, (4.0, 12.0)),
    "Hierarchical triple (stable)": (hierarchical_triple_stars, (12.0, 22.0)),
    "Pythagorean (stress test)": (pythagorean_stars, (4.0, 12.0)),
}

SCENARIO_NOTES = {
    "Original (chaotic)": "Arbitrary start: chaotic, planets scatter quickly.",
    "Figure-8 (periodic)": "Equal masses on one figure-eight; repeats every 1.0068 yr.",
    "Lagrange triangle (unstable)": "Rigid rotating triangle; tiny perturbations break it up.",
    "Hierarchical triple (stable)": "Binary (P = 0.19 yr) + outer star (P = 6.45 yr).",
    "Pythagorean (stress test)": "Close encounters; fixed dt fails by ~3 yr. Watch |dE/E0|.",
}


def build_scenario(name):
    """Full-system (positions, velocities, masses, trinary planet radii) for a scenario."""
    pos, vel, mass = initial_positions.copy(), initial_velocities.copy(), initial_masses.copy()
    planet_radii = planet_orbit_radii[TRINARY_PLANETS].copy()
    if name not in SCENARIO_BUILDERS:
        return pos, vel, mass, planet_radii

    builder, (r_in, r_out) = SCENARIO_BUILDERS[name]
    star_pos, star_vel, star_mass = builder()
    planet_radii = np.geomspace(r_in, r_out, len(TRINARY_PLANETS))
    planet_pos, planet_vel = circular_planet_orbits(planet_radii, float(star_mass.sum()))

    bodies = np.asarray(TRINARY_BODIES, dtype=np.intp)
    old_com_pos = np.average(pos[bodies], axis=0, weights=mass[bodies])
    old_com_vel = np.average(vel[bodies], axis=0, weights=mass[bodies])

    mass[TRINARY_STARS] = star_mass
    pos[bodies] = np.vstack([star_pos, planet_pos])
    vel[bodies] = np.vstack([star_vel, planet_vel])

    # Keep the trinary where it was in the galaxy, then zero total momentum again.
    pos[bodies] += old_com_pos - np.average(pos[bodies], axis=0, weights=mass[bodies])
    vel[bodies] += old_com_vel - np.average(vel[bodies], axis=0, weights=mass[bodies])
    vel -= np.sum(mass[:, None] * vel, axis=0) / np.sum(mass)
    return pos, vel, mass, planet_radii


SCENARIO_NAMES = ["Original (chaotic)", *SCENARIO_BUILDERS]
scenarios = {name: build_scenario(name) for name in SCENARIO_NAMES}
scenario_name = SCENARIO_NAMES[0]


def apply_scenario(name):
    """Load a scenario as the new initial state and restart from it."""
    global scenario_name
    scenario_name = name
    pos, vel, mass, planet_radii = scenarios[name]
    initial_positions[:] = pos
    initial_velocities[:] = vel
    initial_masses[:] = mass
    planet_orbit_radii[TRINARY_PLANETS] = planet_radii
    reset_simulation()
    dpg.set_value("scenario_note", SCENARIO_NOTES[name])


def limit_visual_overlap(centers, radii, floor_radii, gap=0.9):
    """Shrink displayed radii so no two spheres overlap, never below the physical radius."""
    delta = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, np.inf)

    radius_sum = np.maximum(radii[:, None] + radii[None, :], 1e-30)
    pair_scale = np.minimum(1.0, gap * dist / radius_sum)
    body_scale = pair_scale.min(axis=1)

    return np.maximum(radii * body_scale, floor_radii)


# ============================================================
# BODY EDITING (control panel callbacks)
# ============================================================

# property: (minimum, label, unit); the widget tag is f"{property}_{body_index}".
BODY_VALUE_LIMITS = {
    "mass": (MIN_MASS_SOLAR, "Mass", "solar masses"),
    "volume": (MIN_VOLUME_KM3, "Volume", "km³"),
    "size": (MIN_VISUAL_SCALE, "Visual radius multiplier", ""),
    "rotation_period": (MIN_ROTATION_PERIOD_DAYS, "Rotation period", "days"),
}


def body_value(property_name, body_index):
    """Current value of an editable scalar property, in panel units."""
    if property_name == "mass":
        return float(masses[body_index])
    if property_name == "volume":
        return float(volumes_km3[body_index])
    if property_name == "size":
        return float(body_visual_scales[body_index])
    return float(rotation_period_years[body_index] * DAYS_PER_YEAR)


def density_label(body_index):
    return f"Density: {densities[body_index]:.3f} g/cm³"


def set_body_value(sender, app_data, user_data):
    property_name, body_index, component = user_data

    if property_name in BODY_VALUE_LIMITS:
        minimum, label, unit = BODY_VALUE_LIMITS[property_name]
        if not (math.isfinite(app_data) and app_data >= minimum):
            show_status(f"{label} must be finite and at least {minimum:g} {unit}".rstrip() + ".")
            dpg.set_value(sender, body_value(property_name, body_index))
            return

    if property_name == "position":
        positions[body_index, component] = app_data
        forget_history(body_index)
        invalidate_physics_cache()
    elif property_name == "velocity":
        velocities[body_index, component] = app_data
        invalidate_physics_cache()
    elif property_name in ("mass", "volume"):
        if property_name == "mass":
            masses[body_index] = app_data
        else:
            volumes_km3[body_index] = app_data
        update_body_physics(body_index)
        dpg.set_value(f"density_{body_index}", density_label(body_index))
        if property_name == "mass":
            reset_history()
        invalidate_physics_cache()
    elif property_name == "size":
        body_visual_scales[body_index] = app_data
    elif property_name == "rotation_period":
        # Keep the current spin angle continuous across the period change.
        old_period_years = rotation_period_years[body_index]
        new_period_years = app_data / DAYS_PER_YEAR
        current_angle = (
            rotation_phases[body_index] + 2.0 * math.pi * simulation_years / old_period_years
        )
        rotation_period_years[body_index] = new_period_years
        rotation_phases[body_index] = (
            current_angle - 2.0 * math.pi * simulation_years / new_period_years
        ) % (2.0 * math.pi)


def reset_simulation():
    global simulation_years, last_trail_update

    positions[:] = initial_positions
    velocities[:] = initial_velocities
    masses[:] = initial_masses
    volumes_km3[:] = initial_volumes_km3
    body_visual_scales[:] = initial_body_visual_scales
    rotation_period_years[:] = initial_rotation_period_years
    rotation_phases[:] = initial_rotation_phases
    update_body_physics()

    simulation_years = 0.0
    reset_history()
    invalidate_physics_cache()
    set_paused(False)
    show_status("")
    last_trail_update = 0.0  # refresh trails on the next frame

    for i in range(NUM_BODIES):
        for component in range(3):
            dpg.set_value(f"position_{i}_{component}", float(positions[i, component]))
            dpg.set_value(f"velocity_{i}_{component}", float(velocities[i, component]))
        for property_name in BODY_VALUE_LIMITS:
            dpg.set_value(f"{property_name}_{i}", body_value(property_name, i))
        dpg.set_value(f"density_{i}", density_label(i))


def add_body_input(property_name, body_index, label, value_format, step, step_fast):
    dpg.add_input_float(
        tag=f"{property_name}_{body_index}",
        label=label,
        default_value=body_value(property_name, body_index),
        format=value_format,
        step=step,
        step_fast=step_fast,
        min_value=BODY_VALUE_LIMITS[property_name][0],
        min_clamped=True,
        callback=set_body_value,
        user_data=(property_name, body_index, None),
        width=PANEL_INPUT_WIDTH,
    )


def add_body_controls(body_index, body_name):
    with dpg.collapsing_header(label=body_name, default_open=False):
        dpg.add_text("PHYSICS", color=HUD_ACCENT)
        for property_name, heading, values in (
            ("position", "Position (AU)", positions),
            ("velocity", "Velocity (AU/year)", velocities),
        ):
            dpg.add_text(heading)
            for component, axis in enumerate("XYZ"):
                dpg.add_input_float(
                    tag=f"{property_name}_{body_index}_{component}",
                    label=f"{property_name.capitalize()} {axis}",
                    default_value=float(values[body_index, component]),
                    callback=set_body_value,
                    user_data=(property_name, body_index, component),
                    width=PANEL_INPUT_WIDTH,
                )
            dpg.add_separator()

        mass_step = max(body_value("mass", body_index) * 0.01, MIN_MASS_SOLAR)
        add_body_input(
            "mass", body_index, "Mass (solar masses)", "%.3e", mass_step, mass_step * 10.0
        )
        volume = body_value("volume", body_index)
        add_body_input(
            "volume",
            body_index,
            "Volume (km³)",
            "%.3e",
            max(volume * 0.01, MIN_VOLUME_KM3),
            max(volume * 0.1, MIN_VOLUME_KM3),
        )
        dpg.add_text(density_label(body_index), tag=f"density_{body_index}")

        dpg.add_separator()
        dpg.add_text("DISPLAY (visual only)", color=HUD_ACCENT)
        add_body_input("size", body_index, "Visual Radius Multiplier", "%.2f", 0.1, 0.5)
        period_days = body_value("rotation_period", body_index)
        add_body_input(
            "rotation_period",
            body_index,
            "Rotation Period (days)",
            "%.4f",
            max(period_days * 0.01, MIN_ROTATION_PERIOD_DAYS * 0.1),
            max(period_days * 0.1, MIN_ROTATION_PERIOD_DAYS),
        )


# ============================================================
# POSITION HISTORY (light delay and trails)
# Recent steps for all bodies, oldest first. When the buffer is full the
# oldest HISTORY_TRIM_BATCH entries are dropped in one shift.
# ============================================================

OBSERVER = np.array([0.0, 0.0, -100.0])
HISTORY_CAPACITY = TRAIL_LENGTH + HISTORY_TRIM_BATCH
history_times = np.zeros(HISTORY_CAPACITY)
history_positions = np.zeros((HISTORY_CAPACITY, NUM_BODIES, 3))
history_length = 0
simulation_years = 0.0


def record_history():
    global history_length

    if history_length == HISTORY_CAPACITY:
        keep = HISTORY_CAPACITY - HISTORY_TRIM_BATCH
        history_times[:keep] = history_times[HISTORY_TRIM_BATCH:]
        history_positions[:keep] = history_positions[HISTORY_TRIM_BATCH:]
        history_length = keep
    history_times[history_length] = simulation_years
    history_positions[history_length] = positions
    history_length += 1


def reset_history():
    global history_length

    history_times[0] = simulation_years
    history_positions[0] = positions
    history_length = 1


def forget_history(body_index):
    """Collapse one body's history onto its current position (after a teleport)."""
    history_positions[:history_length, body_index] = positions[body_index]


def light_delayed_times():
    """Per body, the time whose light reaches OBSERVER now."""
    return simulation_years - np.linalg.norm(positions - OBSERVER, axis=1) / C_AU_PER_YEAR


def apparent_positions():
    """Positions as seen from OBSERVER (retarded by light travel time), from the history."""
    if history_length < 2:
        return positions.copy()

    times = history_times[:history_length]
    target = light_delayed_times()
    high = np.clip(np.searchsorted(times, target, side="right"), 1, history_length - 1)
    low = high - 1
    span = times[high] - times[low]
    fraction = np.clip((target - times[low]) / np.where(span > 0.0, span, 1.0), 0.0, 1.0)

    bodies = np.arange(NUM_BODIES)
    start = history_positions[low, bodies]
    return start + (history_positions[high, bodies] - start) * fraction[:, None]


reset_history()

# ============================================================
# PROCEDURAL BACKGROUND STAR CHUNKS
# ============================================================

BACKGROUND_CHUNK_SIZE = 200.0
STARS_PER_CHUNK = 24000
BACKGROUND_CHUNK_OFFSETS = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]


def generate_star_chunk(chunk_x, chunk_y, chunk_z):
    chunk_coordinates = (chunk_x, chunk_y, chunk_z)
    seed = (chunk_x * 73856093 ^ chunk_y * 19349663 ^ chunk_z * 83492791) & 0xFFFFFFFF
    chunk_rng = np.random.default_rng(seed)

    origin = np.array(chunk_coordinates, dtype=np.float64) * BACKGROUND_CHUNK_SIZE

    # Most stars are scattered evenly; a quarter form loose local clusters.
    star_positions = origin + chunk_rng.random((STARS_PER_CHUNK, 3)) * BACKGROUND_CHUNK_SIZE

    cluster_count = STARS_PER_CHUNK // 4
    cluster_centers = origin + chunk_rng.random((16, 3)) * BACKGROUND_CHUNK_SIZE
    cluster_ids = chunk_rng.integers(0, len(cluster_centers), size=cluster_count)

    clustered_positions = cluster_centers[cluster_ids] + chunk_rng.normal(
        0.0,
        BACKGROUND_CHUNK_SIZE * 0.045,
        size=(cluster_count, 3),
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
    palette = np.array(
        [
            (0.72, 0.27, 0.12),  # red
            (1.00, 0.55, 0.28),  # orange
            (0.82, 0.88, 1.00),  # white
            (0.52, 0.68, 1.00),  # blue-white
        ],
        dtype=np.float32,
    )

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

    return np.column_stack([star_positions, colors, sizes]).astype(np.float32)


# ============================================================
# NEBULAE (fixed backdrop blobs, raymarched in NEBULA_FRAGMENT_SHADER)
# ============================================================

nebula_rng = np.random.default_rng(42)
nebula_directions = nebula_rng.normal(size=(NUM_NEBULAE, 3))
nebula_directions /= np.linalg.norm(nebula_directions, axis=1, keepdims=True)

nebula_distances = nebula_rng.uniform(12.0, 85.0, size=NUM_NEBULAE)
nebula_positions = nebula_directions * nebula_distances[:, None]
nebula_radii = nebula_rng.uniform(2.0, 5.0, size=NUM_NEBULAE)

nebula_palette = np.array(
    [
        (0.035, 0.012, 0.075),  # violet
        (0.008, 0.045, 0.055),  # teal
        (0.055, 0.014, 0.025),  # rose
        (0.050, 0.032, 0.008),  # amber
    ],
    dtype=np.float32,
)

nebula_colors = nebula_palette[nebula_rng.integers(0, len(nebula_palette), size=NUM_NEBULAE)]

nebula_strengths = nebula_rng.uniform(0.35, 0.8, size=NUM_NEBULAE).astype(np.float32)

nebula_centers_data = np.column_stack(
    [
        nebula_positions,
        nebula_radii,
    ]
).astype(np.float32)

nebula_colors_data = np.column_stack(
    [
        nebula_colors,
        nebula_strengths,
    ]
).astype(np.float32)

# ============================================================
# CAMERA MATRIX HELPERS
# ============================================================


def perspective(fov_degrees, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )


def look_at(eye, target, up):
    forward = target - eye
    forward /= np.linalg.norm(forward)

    side = np.cross(forward, up)
    side /= np.linalg.norm(side)

    up_vector = np.cross(side, forward)

    return np.array(
        [
            [side[0], side[1], side[2], -np.dot(side, eye)],
            [up_vector[0], up_vector[1], up_vector[2], -np.dot(up_vector, eye)],
            [-forward[0], -forward[1], -forward[2], np.dot(forward, eye)],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def gl_matrix(matrix):
    """Column-major float32 bytes for a mat4 uniform."""
    return matrix.T.astype("f4").tobytes()


# ============================================================
# OPENGL INITIALIZATION
# ============================================================

if not glfw.init():
    raise RuntimeError("Could not initialize GLFW.")

glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.SAMPLES, 8)
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)

WINDOW_TITLE = "N-body Star Systems"
window = glfw.create_window(WIDTH, HEIGHT, WINDOW_TITLE, None, None)

if not window:
    glfw.terminate()
    raise RuntimeError("Could not create OpenGL window.")

glfw.make_context_current(window)
glfw.swap_interval(1)

ctx = moderngl.create_context()
ctx.enable(moderngl.PROGRAM_POINT_SIZE)
ctx.enable(moderngl.DEPTH_TEST)
ctx.enable(moderngl.BLEND)
ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

point_program = ctx.program(
    vertex_shader=POINT_VERTEX_SHADER, fragment_shader=POINT_FRAGMENT_SHADER
)
body_program = ctx.program(vertex_shader=BODY_VERTEX_SHADER, fragment_shader=BODY_FRAGMENT_SHADER)
line_program = ctx.program(vertex_shader=LINE_VERTEX_SHADER, fragment_shader=LINE_FRAGMENT_SHADER)
nebula_program = ctx.program(
    vertex_shader=NEBULA_VERTEX_SHADER, fragment_shader=NEBULA_FRAGMENT_SHADER
)
label_program = ctx.program(
    vertex_shader=LABEL_VERTEX_SHADER, fragment_shader=LABEL_FRAGMENT_SHADER
)

nebula_program["nebula_centers"].write(nebula_centers_data.tobytes())
nebula_program["nebula_colors"].write(nebula_colors_data.tobytes())
nebula_vao = ctx.vertex_array(nebula_program, [])
label_program["label_texture"].value = 0
label_vao = ctx.vertex_array(label_program, [])

body_color_array = np.asarray(body_colors, dtype=np.float32)
star_light_data = np.zeros((MAX_STAR_LIGHTS, 4), dtype=np.float32)
star_light_color_data = np.zeros((MAX_STAR_LIGHTS, 4), dtype=np.float32)
star_light_color_data[:NUM_STARS, :3] = body_color_array[:NUM_STARS]
body_program["star_count"].value = NUM_STARS
body_program["star_light_colors"].write(star_light_color_data.tobytes())

# ============================================================
# SYSTEM LABELS (PIL-rendered textures drawn as screen-space quads)
# ============================================================


def load_label_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a bold system font on Windows/macOS/Linux."""
    for font_path in (
        r"C:\Windows\Fonts\arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def make_label_texture(text, font):
    bounds = font.getbbox(text, stroke_width=1)
    image_width = bounds[2] - bounds[0] + 28
    image_height = bounds[3] - bounds[1] + 18

    image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (1, 1, image_width - 2, image_height - 2),
        radius=4,
        fill=(4, 12, 23, 205),
        outline=(45, 205, 220, 220),
        width=1,
    )
    draw.line((7, 3, image_width - 8, 3), fill=(255, 166, 70, 235), width=2)
    draw.text(
        (14 - bounds[0], 9 - bounds[1]),
        text,
        font=font,
        fill=(185, 241, 248, 255),
        stroke_width=1,
        stroke_fill=(2, 8, 16, 230),
    )

    texture = ctx.texture(image.size, 4, image.tobytes(), alignment=1)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return texture, image.size


label_font = load_label_font(18)
system_label_textures = [make_label_texture(name.upper(), label_font) for name in system_names]


def draw_system_labels(projection, view, width, height):
    """Draw each system's name just above its projected centre of mass."""
    ctx.disable(moderngl.DEPTH_TEST)
    view_projection = projection @ view

    for star_indices, (texture, (label_width, label_height)) in zip(
        system_star_indices, system_label_textures, strict=True
    ):
        indices = np.asarray(star_indices, dtype=np.intp)
        center = np.average(positions[indices], axis=0, weights=masses[indices])
        clip = view_projection @ np.append(center, 1.0)
        if clip[3] <= 0.0:
            continue

        ndc = clip[:3] / clip[3]
        if abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0 or not -1.0 <= ndc[2] <= 1.0:
            continue

        label_program["center_ndc"].value = (
            float(ndc[0]),
            float(ndc[1] + (label_height * 0.5 + 12.0) * 2.0 / height),
        )
        label_program["size_ndc"].value = (label_width * 2.0 / width, label_height * 2.0 / height)
        texture.use(location=0)
        label_vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4)

    ctx.enable(moderngl.DEPTH_TEST)


print("OpenGL renderer:", ctx.info.get("GL_RENDERER", "Unknown"))
print(f"Bodies: {NUM_STARS} stars, {NUM_PLANETS} planets in {len(system_names)} systems")
print("Drag mouse to rotate; scroll to zoom; Esc twice to exit.")

# ============================================================
# BACKGROUND STAR BUFFERS
# The 3x3x3 chunks around the camera; one missing chunk is generated per frame.
# ============================================================

background_chunks = {}
pending_background_chunks = []
loaded_background_chunk = None


def load_background_chunks(center_chunk):
    global loaded_background_chunk, pending_background_chunks

    if center_chunk != loaded_background_chunk:
        wanted = {
            tuple(c + o for c, o in zip(center_chunk, offset, strict=True))
            for offset in BACKGROUND_CHUNK_OFFSETS
        }

        for chunk_key in list(background_chunks):
            if chunk_key not in wanted:
                buffer, vao = background_chunks.pop(chunk_key)
                vao.release()
                buffer.release()

        pending_background_chunks = sorted(
            (chunk_key for chunk_key in wanted if chunk_key not in background_chunks),
            key=lambda key: sum((k - c) ** 2 for k, c in zip(key, center_chunk, strict=True)),
        )
        loaded_background_chunk = center_chunk

    if pending_background_chunks:
        chunk_key = pending_background_chunks.pop(0)
        buffer = ctx.buffer(generate_star_chunk(*chunk_key).tobytes())
        vao = ctx.vertex_array(
            point_program,
            [(buffer, "3f 3f 1f", "in_position", "in_color", "in_size")],
        )
        background_chunks[chunk_key] = (buffer, vao)


def chunk_is_visible(chunk_key, forward, right, camera_up, aspect, far_plane):
    """Conservative frustum test for one background chunk."""
    chunk_center = (np.asarray(chunk_key, dtype=np.float64) + 0.5) * BACKGROUND_CHUNK_SIZE
    relative = chunk_center - camera_position

    depth = float(np.dot(relative, forward))
    half_size = BACKGROUND_CHUNK_SIZE * 0.5
    depth_extent = half_size * np.abs(forward).sum()
    horizontal_extent = half_size * np.abs(right).sum()
    vertical_extent = half_size * np.abs(camera_up).sum()

    if depth + depth_extent < 0.1 or depth - depth_extent > far_plane:
        return False

    farthest_depth = max(depth + depth_extent, 0.0)
    half_height = farthest_depth * math.tan(math.radians(FOV_DEGREES) * 0.5)
    half_width = half_height * aspect

    horizontal = abs(float(np.dot(relative, right)))
    vertical = abs(float(np.dot(relative, camera_up)))

    return (
        horizontal <= half_width + horizontal_extent and vertical <= half_height + vertical_extent
    )


# ============================================================
# BODY MESH (one unit sphere, instanced per body)
# ============================================================


def create_sphere_mesh(latitude_segments=128, longitude_segments=256):
    lat = np.linspace(0.0, math.pi, latitude_segments + 1)
    lon = np.linspace(0.0, 2.0 * math.pi, longitude_segments + 1)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    vertices = (
        np.stack(
            [
                np.sin(lat_grid) * np.cos(lon_grid),
                np.cos(lat_grid),
                np.sin(lat_grid) * np.sin(lon_grid),
            ],
            axis=-1,
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )

    row = longitude_segments + 1
    i = np.arange(latitude_segments)[:, None]
    j = np.arange(longitude_segments)[None, :]
    top_left = i * row + j
    bottom_left = (i + 1) * row + j
    top_right = top_left + 1
    bottom_right = bottom_left + 1
    indices = (
        np.stack(
            [top_left, bottom_left, top_right, top_right, bottom_left, bottom_right],
            axis=-1,
        )
        .reshape(-1)
        .astype(np.uint32)
    )
    return vertices, indices


BODY_INSTANCE_FLOATS = 16
sphere_vertices, sphere_indices = create_sphere_mesh()
sphere_buffer = ctx.buffer(sphere_vertices.tobytes())
sphere_index_buffer = ctx.buffer(sphere_indices.tobytes())
body_buffer = ctx.buffer(reserve=NUM_BODIES * BODY_INSTANCE_FLOATS * 4)
body_vao = ctx.vertex_array(
    body_program,
    [
        (sphere_buffer, "3f", "in_unit_position"),
        (
            body_buffer,
            "3f 3f 1f 1f 3f 1f 1f 1f 1f 1f /i",
            "in_center",
            "in_color",
            "in_radius",
            "in_is_star",
            "in_rotation_axis",
            "in_rotation_angle",
            "in_is_gas",
            "in_seed",
            "in_oblateness",
            "in_roughness",
        ),
    ],
    index_buffer=sphere_index_buffer,
    index_element_size=4,
)

# ============================================================
# TRAJECTORY RENDERING
# ============================================================

trail_vaos = [None] * NUM_BODIES
trail_buffers = [None] * NUM_BODIES
trail_vertex_counts = np.zeros(NUM_BODIES, dtype=np.int32)
MAX_TRAIL_VERTICES = 2 * HISTORY_CAPACITY + 1
last_trail_update = 0.0


def visible_body_count():
    """Stars come first, so hiding planets means drawing only the first NUM_STARS bodies."""
    return NUM_BODIES if show_planets else NUM_STARS


def smooth_trail_points(points, samples_per_segment=2):
    """Catmull-Rom resampling of a polyline."""
    if len(points) < 3:
        return points

    p0 = np.vstack([points[0], points[:-2]])
    p1 = points[:-1]
    p2 = points[1:]
    p3 = np.vstack([points[2:], points[-1]])
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False, dtype=np.float32)[None, :, None]

    segments = 0.5 * (
        2.0 * p1[:, None, :]
        + (-p0 + p2)[:, None, :] * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[:, None, :] * t**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[:, None, :] * t**3
    )
    return np.vstack([segments.reshape(-1, 3), points[-1]])


def trim_trail_at_body(points, center, body_radius):
    """Cut the trail where it enters the body's sphere, so it never pokes through."""
    trail_gap = body_radius * 1.03
    distances = np.linalg.norm(points - center, axis=1)
    outside_indices = np.flatnonzero(distances >= trail_gap)
    if len(outside_indices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    last_outside = int(outside_indices[-1])
    if last_outside == len(points) - 1:
        return points

    segment_start = points[last_outside]
    segment = points[last_outside + 1] - segment_start
    relative_start = segment_start - center
    quadratic_a = float(np.dot(segment, segment))
    if quadratic_a < 1e-20:
        return points[: last_outside + 1]

    quadratic_b = 2.0 * float(np.dot(relative_start, segment))
    quadratic_c = float(np.dot(relative_start, relative_start)) - trail_gap**2
    discriminant = max(quadratic_b**2 - 4.0 * quadratic_a * quadratic_c, 0.0)
    intersection_fraction = (-quadratic_b - math.sqrt(discriminant)) / (2.0 * quadratic_a)
    intersection_fraction = float(np.clip(intersection_fraction, 0.0, 1.0))
    intersection = segment_start + intersection_fraction * segment
    intersection = center + (intersection - center) * 1.01

    return np.vstack([points[: last_outside + 1], intersection.astype(np.float32)])


def update_trails(displayed_positions, displayed_radii):
    """Rebuild trail vertex buffers from the light-delayed part of the history."""
    cuts = np.searchsorted(history_times[:history_length], light_delayed_times(), side="right")

    for i in range(visible_body_count()):
        trail_vertex_counts[i] = 0
        if history_length < 2 or cuts[i] < 1:
            continue

        points = np.vstack([history_positions[: cuts[i], i], displayed_positions[i]])
        points = smooth_trail_points(points.astype(np.float32))
        points = trim_trail_at_body(points, displayed_positions[i], float(displayed_radii[i]))
        if len(points) < 2:
            continue

        data = np.empty((len(points), 6), dtype=np.float32)
        data[:, :3] = points
        data[:, 3:] = body_color_array[i]

        if trail_buffers[i] is None:
            trail_buffers[i] = ctx.buffer(reserve=MAX_TRAIL_VERTICES * 6 * 4, dynamic=True)
            trail_vaos[i] = ctx.vertex_array(
                line_program,
                [(trail_buffers[i], "3f 3f", "in_position", "in_color")],
            )

        trail_buffers[i].write(data.tobytes())
        trail_vertex_counts[i] = len(points)


# ============================================================
# FREE-FLY CAMERA CONTROLS
# ============================================================

camera_position = np.array([0.0, 0.0, 4.5], dtype=np.float64)
camera_yaw = -math.pi / 2
camera_pitch = 0.0
# Displayed bodies from the last frame; the camera must stay outside them.
camera_body_positions = positions.copy()
camera_body_radii = collision_radii.copy()

mouse_down = False
last_mouse_x = 0.0
last_mouse_y = 0.0
last_escape_press = -math.inf


def get_camera_forward():
    return np.array(
        [
            math.cos(camera_yaw) * math.cos(camera_pitch),
            math.sin(camera_pitch),
            math.sin(camera_yaw) * math.cos(camera_pitch),
        ],
        dtype=np.float64,
    )


def camera_basis():
    """Forward, right and up unit vectors of the camera."""
    forward = get_camera_forward()
    right = np.cross(forward, WORLD_UP)
    right /= np.linalg.norm(right)
    return forward, right, np.cross(right, forward)


def key_held(key):
    return glfw.get_key(window, key) == glfw.PRESS


def mouse_button_callback(window, button, action, mods):
    global mouse_down, last_mouse_x, last_mouse_y

    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_down = action == glfw.PRESS
        last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)


def sync_mouse_down(window):
    global mouse_down, last_mouse_x, last_mouse_y

    mouse_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)


def window_focus_callback(window, focused):
    global mouse_down

    if focused:
        sync_mouse_down(window)
    else:
        mouse_down = False


def cursor_enter_callback(window, entered):
    if entered:
        sync_mouse_down(window)


def key_callback(window, key, scancode, action, mods):
    global last_escape_press

    if action != glfw.PRESS:
        return
    if key == glfw.KEY_P:
        toggle_pause()
    elif key == glfw.KEY_ESCAPE:
        now = glfw.get_time()
        if now - last_escape_press < ESC_CONFIRM_SECONDS:
            glfw.set_window_should_close(window, True)
        else:
            last_escape_press = now


def cursor_position_callback(window, x, y):
    global camera_yaw, camera_pitch, last_mouse_x, last_mouse_y

    if mouse_down:
        camera_yaw += (x - last_mouse_x) * MOUSE_SENSITIVITY
        camera_pitch -= (y - last_mouse_y) * MOUSE_SENSITIVITY
        camera_pitch = float(np.clip(camera_pitch, -1.5, 1.5))

    last_mouse_x = x
    last_mouse_y = y


def scroll_callback(window, x_offset, y_offset):
    move_camera_by(get_camera_forward() * y_offset * 0.5)


def travel_to_system(system_index):
    """Jump the camera to a viewpoint above and in front of a system."""
    global camera_position, camera_yaw, camera_pitch

    star_indices = np.asarray(system_star_indices[system_index], dtype=np.intp)
    center = np.average(positions[star_indices], axis=0, weights=masses[star_indices])

    largest_orbit = float(np.max(planet_orbit_radii[system_planet_indices[system_index]]))
    viewing_distance = max(12.0, largest_orbit * 5.0)
    camera_position = center + np.array([0.0, viewing_distance * 0.25, viewing_distance])

    direction = center - camera_position
    direction /= np.linalg.norm(direction)
    camera_yaw = math.atan2(direction[2], direction[0])
    camera_pitch = math.asin(float(direction[1]))


def update_camera(delta_time):
    forward, right, _ = camera_basis()
    movement = (
        forward * (key_held(glfw.KEY_W) - key_held(glfw.KEY_S))
        + right * (key_held(glfw.KEY_D) - key_held(glfw.KEY_A))
        + WORLD_UP * (key_held(glfw.KEY_SPACE) - key_held(glfw.KEY_LEFT_CONTROL))
    )
    length = float(np.linalg.norm(movement))
    if length == 0.0:
        constrain_camera_to_bodies(camera_body_positions, camera_body_radii)
        return

    boosted = key_held(glfw.KEY_LEFT_SHIFT) or key_held(glfw.KEY_RIGHT_SHIFT)
    speed = CAMERA_SPEED * (CAMERA_BOOST if boosted else 1.0)
    move_camera_by(movement / length * speed * delta_time)


def move_camera_by(displacement):
    """Move the camera, stopping just short of the first body in the way."""
    global camera_position

    constrain_camera_to_bodies(camera_body_positions, camera_body_radii)
    start = camera_position.copy()
    camera_position = resolve_camera_motion(
        start, start + displacement, camera_body_positions, camera_body_radii
    )


def camera_clearance(radius):
    return radius + max(radius * 0.05, 1e-3)


def constrain_camera_to_bodies(centers, radii):
    """Push the camera out of any visible body it ended up inside."""
    global camera_position

    count = visible_body_count()
    for center, radius in zip(centers[:count], radii[:count], strict=True):
        offset = camera_position - center
        distance = float(np.linalg.norm(offset))
        minimum_distance = camera_clearance(radius)
        if distance >= minimum_distance:
            continue

        if distance < 1e-12:
            offset = -get_camera_forward()
        else:
            offset /= distance
        camera_position = center + offset * minimum_distance


def resolve_camera_motion(start, end, centers, radii):
    """Clip the segment start->end at the first visible body sphere it hits."""
    displacement = end - start
    displacement_squared = float(np.dot(displacement, displacement))
    if displacement_squared < 1e-20:
        return end

    earliest_intersection = 1.0
    displacement_length = math.sqrt(displacement_squared)
    count = visible_body_count()
    for center, radius in zip(centers[:count], radii[:count], strict=True):
        safe_radius = camera_clearance(radius)
        relative_start = start - center
        quadratic_c = float(np.dot(relative_start, relative_start)) - safe_radius**2
        quadratic_b = 2.0 * float(np.dot(relative_start, displacement))
        if quadratic_c <= 1e-10:
            if quadratic_b < 0.0:
                earliest_intersection = 0.0
            continue

        discriminant = quadratic_b**2 - 4.0 * displacement_squared * quadratic_c
        if discriminant < 0.0:
            continue

        intersection = (-quadratic_b - math.sqrt(discriminant)) / (2.0 * displacement_squared)
        if 0.0 <= intersection < earliest_intersection:
            earliest_intersection = intersection

    if earliest_intersection < 1.0:
        safe_fraction = max(0.0, earliest_intersection - 1e-3 / displacement_length)
        return start + displacement * safe_fraction
    return end


glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_position_callback)
glfw.set_scroll_callback(window, scroll_callback)
glfw.set_window_focus_callback(window, window_focus_callback)
glfw.set_cursor_enter_callback(window, cursor_enter_callback)
glfw.set_key_callback(window, key_callback)


# ============================================================
# CONTROL PANEL
# A separate DearPyGui viewport, rendered from the main loop (macOS only
# allows windows on the main thread). Callbacks are queued by DPG and run in
# render_control_panel_frame, so they never race the physics.
# ============================================================


def set_paused(value):
    global paused

    paused = value
    if control_panel_open:
        dpg.set_item_label("pause_button", "Resume (P)" if paused else "Pause (P)")


def toggle_pause():
    set_paused(not paused)


def show_status(message):
    if message:
        print(message)
    if control_panel_open:
        dpg.set_value("status_text", message)


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


def build_theme():
    with dpg.theme() as hud_theme, dpg.theme_component(dpg.mvAll):
        for name, color in HUD_COLORS.items():
            dpg.add_theme_color(getattr(dpg, f"mvThemeCol_{name}"), color)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 5)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3)
    dpg.bind_theme(hud_theme)


def build_font():
    """Bind a TTF panel font; on macOS rasterise at retina scale and draw it back down."""
    font_path = next((path for path in PANEL_FONT_PATHS if os.path.exists(path)), None)
    if font_path is None:
        return
    content_scale = max(glfw.get_window_content_scale(window)[0], 1.0)
    with dpg.font_registry():
        font = dpg.add_font(font_path, round(PANEL_FONT_SIZE * content_scale))
    dpg.bind_font(font)
    if sys.platform == "darwin":
        dpg.set_global_font_scale(1.0 / content_scale)


def build_global_controls():
    dpg.add_button(label="Reset Simulation", callback=reset_simulation)
    dpg.add_button(label="Pause (P)", tag="pause_button", callback=toggle_pause)
    dpg.add_text("", tag="status_text", color=HUD_ACCENT_ACTIVE, wrap=0)
    dpg.add_text(
        "Camera: drag left mouse to look, W/A/S/D move, Space up, Ctrl down, "
        "hold Shift to move faster, scroll to zoom. P pause, Esc twice to quit.",
        wrap=0,
    )
    dpg.add_separator()

    dpg.add_combo(
        SCENARIO_NAMES,
        label="Scenario",
        default_value=scenario_name,
        width=PANEL_INPUT_WIDTH,
        callback=lambda s, a: apply_scenario(a),
    )
    dpg.add_text(SCENARIO_NOTES[scenario_name], tag="scenario_note", wrap=0)
    dpg.add_combo(
        list(INTEGRATORS),
        label="Integrator",
        default_value=integrator_name,
        width=PANEL_INPUT_WIDTH,
        callback=lambda s, a: set_integrator(a),
    )
    dpg.add_combo(
        TIMESTEP_OPTIONS,
        label="Timestep dt (yr)",
        default_value=f"{DT:.0e}",
        width=PANEL_INPUT_WIDTH,
        callback=lambda s, a: set_timestep(a),
    )
    dpg.add_slider_float(
        label="Time Speed",
        min_value=0.1,
        max_value=10.0,
        default_value=time_scale,
        width=PANEL_INPUT_WIDTH,
        callback=lambda s, a: set_time_scale(a),
    )
    dpg.add_separator()

    dpg.add_text("DISPLAY (visual only, physics unchanged)")
    dpg.add_checkbox(
        label="Show Trajectories",
        default_value=show_trails,
        callback=lambda s, a: set_show_trails(a),
    )
    dpg.add_checkbox(
        label="Show Planets",
        default_value=show_planets,
        callback=lambda s, a: set_show_planets(a),
    )
    dpg.add_slider_float(
        label="Star Size Exaggeration",
        min_value=0.25,
        max_value=4.0,
        default_value=star_scale,
        width=PANEL_INPUT_WIDTH,
        callback=lambda s, a: set_star_scale(a),
    )
    dpg.add_slider_float(
        label="Planet Size Exaggeration",
        min_value=0.25,
        max_value=4.0,
        default_value=planet_scale,
        width=PANEL_INPUT_WIDTH,
        callback=lambda s, a: set_planet_scale(a),
    )
    dpg.add_separator()


def build_diagnostics():
    with dpg.collapsing_header(label="Conservation diagnostics", default_open=False):
        dpg.add_text(
            "Relative drift of conserved quantities since the baseline. "
            f"Energy: below {ENERGY_ERROR_GOOD:.0e} good, above {ENERGY_ERROR_POOR:.0e} "
            "means the timestep is too large for this configuration.",
            wrap=0,
        )
        dpg.add_text("", tag="diag_time")
        dpg.add_text("", tag="diag_energy")
        dpg.add_text("", tag="diag_energy_error")
        dpg.add_text("", tag="diag_momentum")
        dpg.add_text("", tag="diag_angular_error")
        dpg.add_text(
            "Baseline resets after Reset, any body edit, or an integrator/timestep change.",
            wrap=0,
        )
        with dpg.plot(label="Relative energy error", height=180, width=-1):
            dpg.add_plot_axis(dpg.mvXAxis, label="time (yr)", tag="energy_x_axis")
            with dpg.plot_axis(dpg.mvYAxis, label="log10 |dE/E0|", tag="energy_y_axis"):
                dpg.add_line_series([], [], tag="energy_series")
    dpg.add_separator()


def build_system_sections():
    dpg.add_text("STAR SYSTEMS")
    for system_index, system_name in enumerate(system_names):
        with dpg.collapsing_header(label=system_name, default_open=(system_index == 0)):
            dpg.add_button(
                label="Travel to system",
                callback=lambda s, a, u: travel_to_system(u),
                user_data=system_index,
            )
            for star_index in system_star_indices[system_index]:
                add_body_controls(star_index, body_names[star_index])
            for planet_index in system_planet_indices[system_index]:
                body_index = NUM_STARS + planet_index
                add_body_controls(body_index, body_names[body_index])


def run_control_panel():
    global control_panel_open

    dpg.create_context()
    dpg.configure_app(manual_callback_management=True)
    build_theme()
    build_font()

    with dpg.window(label="Simulation Controls", tag="control_window"):
        build_global_controls()
        build_diagnostics()
        build_system_sections()

    dpg.create_viewport(title="Simulation Controls", width=470, height=850, vsync=False)
    dpg.set_primary_window("control_window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    glfw.make_context_current(window)
    control_panel_open = True


def render_control_panel_frame():
    """Run queued panel callbacks, then draw one panel frame."""
    global control_panel_open

    if not control_panel_open:
        return
    if not dpg.is_dearpygui_running():
        control_panel_open = False
        return

    dpg.run_callbacks(dpg.get_callback_queue())
    dpg.render_dearpygui_frame()
    glfw.make_context_current(window)


# ============================================================
# FRAME STEPS
# ============================================================


def advance_simulation():
    """Run this frame's physics steps; returns the number of steps attempted."""
    global simulation_years

    if paused:
        return 0
    steps = max(1, int(STEPS_PER_FRAME * time_scale))
    for _ in range(steps):
        if not physics_step():
            break
        simulation_years += DT
        record_history()
    return steps


def displayed_state():
    """Light-delayed positions and exaggerated, non-overlapping radii for rendering."""
    shown_positions = apparent_positions()
    shown_radii = collision_radii * body_visual_exaggeration * body_visual_scales
    shown_radii[:NUM_STARS] *= star_scale
    shown_radii[NUM_STARS:] *= planet_scale

    count = visible_body_count()
    shown_radii[:count] = limit_visual_overlap(
        shown_positions[:count], shown_radii[:count], collision_radii[:count]
    )
    return shown_positions, shown_radii


def refresh_trails(shown_positions, shown_radii, steps, delta_time):
    """Rebuild trails every TRAIL_UPDATE_INTERVAL; cut them further out at high speed."""
    global last_trail_update

    now = time.perf_counter()
    interval = TRAIL_UPDATE_INTERVAL / max(time_scale, 1.0)
    if now - last_trail_update < interval:
        return

    # A body moves this far before the next refresh; keep the trail tip out of its sphere.
    simulation_interval = DT * steps * interval / delta_time
    cut_radii = shown_radii + np.linalg.norm(velocities, axis=1) * simulation_interval
    update_trails(shown_positions, cut_radii)
    last_trail_update = now


def write_body_instances(shown_positions, shown_radii):
    rotation_angles = (
        rotation_phases + 2.0 * math.pi * simulation_years / rotation_period_years
    ) % (2.0 * math.pi)

    star_light_data[:NUM_STARS, :3] = shown_positions[:NUM_STARS]
    # Mass-luminosity relation L ~ M^3.5.
    star_light_data[:NUM_STARS, 3] = np.maximum(masses[:NUM_STARS], MIN_MASS_SOLAR) ** 3.5

    body_data = np.hstack(
        [
            shown_positions,
            body_color_array,
            shown_radii[:, None],
            body_is_star[:, None],
            rotation_axes,
            rotation_angles[:, None],
            body_is_gas[:, None],
            body_seeds[:, None],
            body_oblateness[:, None],
            body_roughness[:, None],
        ]
    ).astype(np.float32)
    body_buffer.write(body_data.tobytes())


def camera_matrices(shown_positions, shown_radii, aspect):
    """Projection and view; the near plane shrinks as the camera nears a surface."""
    body_distances = np.linalg.norm(shown_positions - camera_position, axis=1)
    far_plane = max(200.0, float(np.max(body_distances + shown_radii)) + 100.0)

    count = visible_body_count()
    nearest_surface = max(float(np.min(body_distances[:count] - shown_radii[:count])), 1e-5)
    near_plane = max(1e-5, min(0.01, nearest_surface * 0.25))

    projection = perspective(FOV_DEGREES, aspect, near_plane, far_plane)
    view = look_at(camera_position, camera_position + get_camera_forward(), WORLD_UP)
    return projection, view, far_plane


def draw_background(projection, view, width, height, far_plane):
    nebula_program["resolution"].value = (float(width), float(height))
    nebula_program["inverse_projection"].write(gl_matrix(np.linalg.inv(projection)))
    nebula_program["inverse_view"].write(gl_matrix(np.linalg.inv(view)))
    nebula_program["camera_position"].value = tuple(float(v) for v in camera_position)
    nebula_vao.render(mode=moderngl.TRIANGLES, vertices=3)

    current_chunk = tuple(int(c) for c in np.floor(camera_position / BACKGROUND_CHUNK_SIZE))
    load_background_chunks(current_chunk)
    forward, right, camera_up = camera_basis()
    for chunk_key, (_buffer, vao) in background_chunks.items():
        if chunk_is_visible(chunk_key, forward, right, camera_up, width / height, far_plane):
            vao.render(mode=moderngl.POINTS)


def draw_scene(projection, view, far_plane, width, height, current_time):
    mvp = gl_matrix(projection @ view)
    point_program["mvp"].write(mvp)
    line_program["mvp"].write(mvp)
    body_program["view"].write(gl_matrix(view))
    body_program["projection"].write(gl_matrix(projection))
    body_program["star_lights"].write(star_light_data.tobytes())
    body_program["camera_position"].value = tuple(float(v) for v in camera_position)
    body_program["time"].value = float(current_time)

    ctx.viewport = (0, 0, width, height)
    ctx.clear(0.005, 0.007, 0.02, 1.0)
    draw_background(projection, view, width, height, far_plane)

    count = visible_body_count()
    body_vao.render(mode=moderngl.TRIANGLES, vertices=len(sphere_indices), instances=count)

    # Trails test against the finished body depth buffer without writing to it.
    if show_trails:
        ctx.depth_mask = False
        try:
            for vao, vertex_count in zip(
                trail_vaos[:count], trail_vertex_counts[:count], strict=True
            ):
                if vao is not None and vertex_count > 1:
                    vao.render(mode=moderngl.LINE_STRIP, vertices=int(vertex_count))
        finally:
            ctx.depth_mask = True

    draw_system_labels(projection, view, width, height)


def update_window_title(now):
    global last_escape_press

    if now - last_escape_press >= ESC_CONFIRM_SECONDS:
        last_escape_press = -math.inf
    escape_hint = " - press Esc again to quit" if math.isfinite(last_escape_press) else ""
    glfw.set_window_title(
        window,
        f"{WINDOW_TITLE} | {NUM_STARS} stars, {NUM_PLANETS} planets"
        f" | t = {simulation_years:.5f} yr{escape_hint}",
    )


def release_gl_resources():
    for texture, _ in system_label_textures:
        texture.release()
    for gl_object in [*trail_vaos, *trail_buffers]:
        if gl_object is not None:
            gl_object.release()
    for buffer, vao in background_chunks.values():
        vao.release()
        buffer.release()
    for gl_object in (
        label_vao,
        nebula_vao,
        body_vao,
        body_buffer,
        sphere_index_buffer,
        sphere_buffer,
        label_program,
        nebula_program,
        point_program,
        line_program,
        body_program,
    ):
        gl_object.release()
    ctx.release()


# ============================================================
# MAIN LOOP
# ============================================================

run_control_panel()
last_frame_time = glfw.get_time()

try:
    while not glfw.window_should_close(window):
        glfw.poll_events()
        render_control_panel_frame()
        current_time = glfw.get_time()
        delta_time = max(current_time - last_frame_time, 1e-6)
        last_frame_time = current_time

        steps = advance_simulation()
        update_diagnostics()

        shown_positions, shown_radii = displayed_state()
        camera_body_positions[:] = shown_positions
        camera_body_radii[:] = shown_radii
        refresh_trails(shown_positions, shown_radii, steps, delta_time)
        write_body_instances(shown_positions, shown_radii)

        update_camera(delta_time)
        constrain_camera_to_bodies(shown_positions, shown_radii)

        width, height = (max(size, 1) for size in glfw.get_framebuffer_size(window))
        projection, view, far_plane = camera_matrices(shown_positions, shown_radii, width / height)
        draw_scene(projection, view, far_plane, width, height, current_time)
        glfw.swap_buffers(window)
        update_window_title(current_time)
finally:
    release_gl_resources()
    dpg.destroy_context()
    glfw.destroy_window(window)
    glfw.terminate()
