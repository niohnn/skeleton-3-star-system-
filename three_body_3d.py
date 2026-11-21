import math
import time
from queue import SimpleQueue

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
BODY_NOISE_GLSL = """
float hash3(vec3 point) {
    return fract(sin(dot(point, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

float value_noise3(vec3 point) {
    vec3 cell = floor(point);
    vec3 f = fract(point);
    f = f * f * (3.0 - 2.0 * f);

    float x00 = mix(hash3(cell), hash3(cell + vec3(1.0, 0.0, 0.0)), f.x);
    float x10 = mix(hash3(cell + vec3(0.0, 1.0, 0.0)), hash3(cell + vec3(1.0, 1.0, 0.0)), f.x);
    float x01 = mix(hash3(cell + vec3(0.0, 0.0, 1.0)), hash3(cell + vec3(1.0, 0.0, 1.0)), f.x);
    float x11 = mix(hash3(cell + vec3(0.0, 1.0, 1.0)), hash3(cell + vec3(1.0, 1.0, 1.0)), f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

float fbm3(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 4; i++) {
        value += amplitude * value_noise3(p);
        p = p * 2.0 + vec3(13.7, 9.2, 5.3);
        amplitude *= 0.5;
    }
    return value / 0.9375;
}

// Shared by the vertex shader (shape) and fragment shader (colour),
// so mountains and colours line up.
float terrain_height(vec3 p, float seed) {
    return fbm3(p * 1.5 + vec3(seed));
}

vec3 rotate_around_axis(vec3 v, vec3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}
"""

BODY_VERTEX_SHADER = (
    "#version 330\n"
    + BODY_NOISE_GLSL
    + """
in vec3 in_unit_position;
in vec3 in_center;
in vec3 in_color;
in float in_radius;
in float in_is_star;
in vec3 in_rotation_axis;
in float in_rotation_angle;
in float in_is_gas;
in float in_seed;
in float in_oblateness;
in float in_roughness;

uniform mat4 view;
uniform mat4 projection;

out vec3 vertex_color;
out vec3 world_position;
out vec3 world_normal;
out vec3 surface_coordinates;
flat out float body_is_star;
flat out float body_is_gas;
flat out float body_seed;
flat out float body_angle;
flat out vec3 body_rotation_axis;

vec3 displaced(vec3 p, float seed, float rough) {
    return p * (1.0 + rough * (terrain_height(p, seed) - 0.5) * 2.0);
}

vec3 terrain_normal(vec3 p, float seed, float rough) {
    vec3 helper = abs(p.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t1 = normalize(cross(p, helper));
    vec3 t2 = cross(p, t1);
    float e = 0.004;
    vec3 q0 = displaced(p, seed, rough);
    vec3 q1 = displaced(normalize(p + t1 * e), seed, rough);
    vec3 q2 = displaced(normalize(p + t2 * e), seed, rough);
    vec3 n = normalize(cross(q1 - q0, q2 - q0));
    return dot(n, p) < 0.0 ? -n : n;
}

void main() {
    vec3 axis = normalize(in_rotation_axis);

    vec3 p = in_unit_position;   // body-fixed coordinates
    vec3 n = p;                  // exact normal on a unit sphere

    bool rocky = in_is_star < 0.5 && in_is_gas < 0.5;
    if (rocky && in_roughness > 0.0) {
        n = terrain_normal(p, in_seed, in_roughness);
        p = displaced(p, in_seed, in_roughness);
    }

    vec3 rp = rotate_around_axis(p, axis, in_rotation_angle);
    vec3 rn = rotate_around_axis(n, axis, in_rotation_angle);

    // Oblateness: squash along the spin axis (normals use the inverse transpose).
    float f = in_oblateness;
    rp -= axis * dot(axis, rp) * f;
    rn = normalize(rn + axis * dot(axis, rn) * (1.0 / (1.0 - f) - 1.0));

    world_position = in_center + rp * in_radius;
    world_normal = rn;

    // Body-fixed coordinates: surface detail now rotates WITH the body.
    surface_coordinates = in_unit_position;

    vertex_color = in_color;
    body_is_star = in_is_star;
    body_is_gas = in_is_gas;
    body_seed = in_seed;
    body_angle = in_rotation_angle;
    body_rotation_axis = axis;

    gl_Position = projection * view * vec4(world_position, 1.0);
}
"""
)

BODY_FRAGMENT_SHADER = (
    "#version 330\n"
    + BODY_NOISE_GLSL
    + """
#define MAX_STAR_LIGHTS 32

in vec3 vertex_color;
in vec3 world_position;
in vec3 world_normal;
in vec3 surface_coordinates;
flat in float body_is_star;
flat in float body_is_gas;
flat in float body_seed;
flat in float body_angle;
flat in vec3 body_rotation_axis;

uniform int star_count;
uniform vec4 star_lights[MAX_STAR_LIGHTS];
uniform vec4 star_light_colors[MAX_STAR_LIGHTS];
uniform vec3 camera_position;
uniform float time;

out vec4 frag_color;

vec3 hue_shift(vec3 c, float a) {
    const vec3 k = vec3(0.57735027);
    float ca = cos(a);
    float sa = sin(a);
    return c * ca + cross(k, c) * sa + k * dot(k, c) * (1.0 - ca);
}

// Cheap bump mapping from the noise gradient. Tune `strength` to taste.
vec3 bump_normal(vec3 n, vec3 s, float scale, float strength) {
    float e = 0.01;
    vec3 sp = s * scale;
    vec3 g = vec3(
        value_noise3(sp + vec3(e, 0.0, 0.0)) - value_noise3(sp - vec3(e, 0.0, 0.0)),
        value_noise3(sp + vec3(0.0, e, 0.0)) - value_noise3(sp - vec3(0.0, e, 0.0)),
        value_noise3(sp + vec3(0.0, 0.0, e)) - value_noise3(sp - vec3(0.0, 0.0, e))
    ) / (2.0 * e);
    g = rotate_around_axis(g, body_rotation_axis, body_angle);
    g -= n * dot(g, n);
    return normalize(n - g * strength);
}

void main() {
    vec3 n = normalize(world_normal);
    vec3 s = normalize(surface_coordinates);
    vec3 axis = normalize(body_rotation_axis);
    vec3 view_dir = normalize(camera_position - world_position);
    float mu = max(dot(n, view_dir), 0.0);
    float seed = body_seed;
    vec3 albedo = vertex_color;

    // ---------------- STARS ----------------
    if (body_is_star > 0.5) {
        float t = time * 0.15;
        float gran = 0.6 * value_noise3(s * 30.0 + vec3(t))
                   + 0.4 * value_noise3(s * 70.0 - vec3(t * 1.7));
        float spots = smoothstep(0.72, 0.88, value_noise3(s * 7.0 + vec3(seed)));
        vec3 hot = albedo * (0.8 + 0.5 * gran) * (1.0 - 0.35 * spots);

        // Limb darkening: edges are cooler and redder.
        float limb = pow(mu, 0.55);
        vec3 limb_tint = mix(vec3(1.0, 0.55, 0.30), vec3(1.0), limb);
        frag_color = vec4(hot * limb_tint * mix(0.55, 1.15, limb), 1.0);
        return;
    }

    // Per-planet hue jitter on top of the base colour (about +/-40 degrees).
    float shift = fract(seed * 12.9898) - 0.5;
    albedo = hue_shift(albedo, (fract(seed * 2.31) - 0.5) * 1.4);

    vec3 bn = n;
    float gloss = 0.0;
    vec3 atmosphere = vec3(0.0);
    float atmosphere_strength = 0.0;

    if (body_is_gas > 0.5) {
        // ---------------- GAS GIANTS ----------------
        float lat = dot(s, axis);
        float warp = fbm3(s * 3.0 + vec3(seed));
        float coarse = value_noise3(vec3(lat * 6.0 + warp * 1.4, seed, 1.0));
        float fine = value_noise3(vec3(lat * 17.0 + warp * 2.2, seed + 4.0, 2.0));

        vec3 band_a = albedo;
        vec3 band_b = hue_shift(albedo, 0.45 + fract(seed * 5.1) * 0.6) * 0.65;
        vec3 band_c = mix(albedo, vec3(1.0), 0.45);

        vec3 gas = mix(band_a, band_b, smoothstep(0.35, 0.65, coarse));
        gas = mix(gas, band_c, smoothstep(0.55, 0.85, fine) * 0.7);

        float storm = smoothstep(0.66, 0.80, value_noise3(s * vec3(4.0, 8.0, 4.0) + vec3(seed)));
        storm *= 1.0 - abs(lat);
        gas = mix(gas, hue_shift(albedo, 2.6) * 1.15, storm * 0.6);

        albedo = gas;
        atmosphere = mix(gas, vec3(0.75, 0.88, 1.0), 0.35);
        atmosphere_strength = 0.8;
    } else {
        // ---------------- ROCKY PLANETS ----------------
        float h = terrain_height(s, seed);
        float detail = fbm3(s * 14.0 + vec3(seed * 1.7));

        vec3 lowland = hue_shift(albedo, shift * 1.2) * 0.45;
        vec3 highland = hue_shift(albedo, -shift * 0.8) * 1.35 + 0.05;
        vec3 land = mix(lowland, albedo, smoothstep(0.34, 0.50, h));
        land = mix(land, highland, smoothstep(0.54, 0.66, h));
        land *= 0.8 + 0.4 * detail;

        bool ocean_world = fract(seed * 3.7) > 0.65;
        float water = ocean_world
            ? 1.0 - smoothstep(0.455, 0.485, h)
            : 0.0;
        vec3 water_color = hue_shift(vec3(0.02, 0.10, 0.26), shift * 0.6);
        albedo = mix(land, water_color, water);
        gloss = water;

        // Polar caps on some worlds.
        float lat = abs(dot(s, axis));
        float cap = smoothstep(0.80, 0.90, lat + (detail - 0.5) * 0.18);
        cap *= step(0.35, fract(seed * 7.31));
        albedo = mix(albedo, vec3(0.92, 0.96, 1.0), cap);
        gloss *= 1.0 - cap;

        bn = normalize(mix(bump_normal(n, s, 16.0, 0.10), n, water));

        bool has_atmosphere = ocean_world || fract(seed * 9.1) > 0.45;
        atmosphere = hue_shift(vec3(0.35, 0.55, 1.0), shift * 1.5);
        atmosphere_strength = has_atmosphere ? 0.7 : 0.0;
    }

    // ---------------- LIGHTING ----------------
    vec3 direct_light = vec3(0.0);
    vec3 spec = vec3(0.0);
    for (int i = 0; i < MAX_STAR_LIGHTS; i++) {
        if (i >= star_count) {
            break;
        }
        vec3 to_light = star_lights[i].xyz - world_position;
        float d2 = max(dot(to_light, to_light), 1e-8);
        vec3 ld = to_light * inversesqrt(d2);
        float irr = star_lights[i].w / d2;
        vec3 lc = star_light_colors[i].rgb;

        direct_light += lc * max(dot(bn, ld), 0.0) * irr;

        vec3 hv = normalize(ld + view_dir);
        spec += lc * pow(max(dot(bn, hv), 0.0), 90.0) * irr
              * step(0.0, dot(n, ld));
    }

    float irradiance = dot(direct_light, vec3(0.2126, 0.7152, 0.0722));
    vec3 light_tint = direct_light / max(irradiance, 1e-6);
    float daylight = 1.0 - exp(-0.75 * irradiance);
    vec3 lighting = vec3(0.018) + light_tint * daylight * 1.2;

    float spec_l = dot(spec, vec3(0.2126, 0.7152, 0.0722));
    vec3 spec_color = light_tint * (1.0 - exp(-0.75 * spec_l)) * gloss * 0.8;

    // Atmospheric rim glow, strongest on the lit side.
    float rim = pow(1.0 - mu, 3.0);
    vec3 rim_light = atmosphere * rim * atmosphere_strength * (0.06 + daylight) * 1.2;

    frag_color = vec4(albedo * lighting + spec_color + rim_light, 1.0);
}

"""
)
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
star_densities = np.array(
    [
        1.41,  # Star 1
        1.41,  # Star 2
        1.41,  # Star 3
    ],
    dtype=np.float64,
)

star_positions = np.array(
    [
        [-0.8, 0.0, 0.3],
        [0.8, -0.3, -0.2],
        [0.1, 1.0, 0.7],
    ],
    dtype=np.float64,
)

star_velocities = np.array(
    [
        [-1.5, -2.5, 1.2],
        [2.5, 1.0, -1.5],
        [-1.0, 0.8, 1.0],
    ],
    dtype=np.float64,
)

# ============================================================
# SIX ROCKY PLANETS + FOUR GAS GIANTS
# Mass: solar masses
# Density: g/cm^3
# Orbital radius: AU
# ============================================================

planet_names = [
    "Rock 1",
    "Rock 2",
    "Rock 3",
    "Rock 4",
    "Rock 5",
    "Rock 6",
    "Gas 1",
    "Gas 2",
    "Gas 3",
    "Gas 4",
]

planet_types = ["rocky"] * 6 + ["gas"] * 4
planet_rotation_period_hours = [
    24.0,
    36.0,
    23.93,
    24.62,
    18.0,
    30.0,
    9.93,
    10.7,
    9.9,
    16.1,
]

planet_masses = np.array(
    [
        1.0e-6,
        2.0e-6,
        3.0e-6,
        2.5e-6,
        4.0e-6,
        5.0e-6,
        1.0e-4,
        2.0e-4,
        3.0e-4,
        5.0e-4,
    ],
    dtype=np.float64,
)

planet_densities = np.array(
    [
        5.5,
        5.2,
        4.5,
        5.0,
        3.9,
        4.0,
        1.3,
        0.8,
        0.5,
        0.3,
    ],
    dtype=np.float64,
)

planet_orbit_radii = np.array(
    [
        0.12,
        0.18,
        0.25,
        0.34,
        0.44,
        0.56,
        0.72,
        0.88,
        1.05,
        1.22,
    ],
    dtype=np.float64,
)

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

    radial = np.array(
        [
            np.cos(angle),
            np.sin(angle) * np.cos(inclination),
            np.sin(angle) * np.sin(inclination),
        ]
    )

    tangent = np.array(
        [
            -np.sin(angle),
            np.cos(angle) * np.cos(inclination),
            np.cos(angle) * np.sin(inclination),
        ]
    )

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

    distance = 400.0 if is_major_leagues else system_rng.uniform(25.0, 200.0)
    system_position = central_position + direction * distance
    system_velocity = central_velocity + tangent * np.sqrt(G * central_mass / distance)

    star_mass = (
        2.0 if is_major_leagues else float(np.exp(system_rng.uniform(np.log(0.2), np.log(1.4))))
    )
    companion_mass = 1.6 if is_major_leagues else 0.0
    planet_central_mass = star_mass + companion_mass

    first_star_index = len(star_masses) + len(extra_star_masses)

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

    extra_star_masses.append(star_mass)
    extra_star_densities.append(1.41 * star_mass**-1.4)
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
        100.0 if is_major_leagues else float(np.clip(42.0 * star_mass**0.8, 16, 48))
    )

    if is_major_leagues:
        extra_star_masses.append(companion_mass)
        extra_star_densities.append(1.41 * companion_mass**-1.4)
        extra_star_positions.append(
            system_position + (star_mass / planet_central_mass) * binary_separation * tangent
        )
        extra_star_velocities.append(
            system_velocity + (star_mass / planet_central_mass) * relative_speed * binary_normal
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
            planet_masses = np.append(planet_masses, system_rng.uniform(2e-5, 5e-4))
            planet_densities = np.append(planet_densities, system_rng.uniform(0.4, 1.8))
            planet_colors.append(gas_colors[int(system_rng.integers(len(gas_colors)))])
            extra_planet_sizes.append(float(system_rng.uniform(8.0, 14.0)))
            planet_rotation_period_hours.append(10.5)
        else:
            planet_masses = np.append(planet_masses, system_rng.uniform(1e-6, 2e-5))
            planet_densities = np.append(planet_densities, system_rng.uniform(3.0, 7.0))
            planet_colors.append(rocky_colors[int(system_rng.integers(len(rocky_colors)))])
            extra_planet_sizes.append(float(system_rng.uniform(5.0, 9.0)))
            planet_rotation_period_hours.append(24.0)

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

        planet_positions.append(system_position + orbit_radius * radial)
        planet_velocities.append(
            system_velocity + np.sqrt(G * planet_central_mass / orbit_radius) * orbital_tangent
        )

    system_planet_indices.append(this_system_planets)

star_masses = np.concatenate(
    [
        star_masses,
        np.asarray(extra_star_masses, dtype=np.float64),
    ]
)
star_densities = np.concatenate(
    [
        star_densities,
        np.asarray(extra_star_densities, dtype=np.float64),
    ]
)
star_positions = np.vstack(
    [
        star_positions,
        np.asarray(extra_star_positions, dtype=np.float64),
    ]
)
star_velocities = np.vstack(
    [
        star_velocities,
        np.asarray(extra_star_velocities, dtype=np.float64),
    ]
)

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
velocities -= np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)

rotation_rng = np.random.default_rng(20261119)
star_rotation_period_years = np.clip(25.0 / np.sqrt(star_masses), 5.0, 60.0) / 365.25
planet_rotation_period_years = np.asarray(planet_rotation_period_hours, dtype=np.float64) / (
    24.0 * 365.25
)
rotation_period_years = np.concatenate(
    [
        star_rotation_period_years,
        planet_rotation_period_years,
    ]
)
rotation_phases = rotation_rng.uniform(0.0, 2.0 * math.pi, NUM_BODIES)
initial_rotation_phases = rotation_phases.copy()
rotation_axes = np.zeros((NUM_BODIES, 3), dtype=np.float64)

for star_index in range(NUM_STARS):
    axis = rotation_rng.normal(size=3)
    rotation_axes[star_index] = axis / np.linalg.norm(axis)

for system_index, planet_indices in enumerate(system_planet_indices):
    star_indices = np.asarray(
        system_star_indices[system_index],
        dtype=np.intp,
    )
    system_center = np.average(
        positions[star_indices],
        axis=0,
        weights=masses[star_indices],
    )
    system_velocity = np.average(
        velocities[star_indices],
        axis=0,
        weights=masses[star_indices],
    )

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

initial_rotation_period_years = rotation_period_years.copy()

# ============================================================
# DISPLAY SETTINGS
# Display exaggeration keeps physical-size spheres visible at astronomical scales.
# ============================================================

star_colors = [
    (1.0, 0.82, 0.48),
    (0.55, 0.72, 1.0),
    (1.0, 0.45, 0.28),
] + extra_star_colors

body_colors = star_colors + planet_colors

body_reference_pixel_sizes = np.array(
    [42, 38, 48] + extra_star_sizes + [5, 5, 6, 6, 6, 7, 9, 10, 11, 12] + extra_planet_sizes,
    dtype=np.float32,
)
body_visual_scales = np.ones(NUM_BODIES, dtype=np.float32)
body_is_star = np.zeros(NUM_BODIES, dtype=np.float32)
body_is_star[:NUM_STARS] = 1.0
body_is_gas = np.zeros(NUM_BODIES, dtype=np.float32)
body_is_gas[NUM_STARS:] = np.asarray(
    [planet_type == "gas" for planet_type in planet_types],
    dtype=np.float32,
)
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
MAX_STAR_LIGHTS = 32
if NUM_STARS > MAX_STAR_LIGHTS:
    raise ValueError(f"At most {MAX_STAR_LIGHTS} stars can be used as shader light sources.")
initial_positions = positions.copy()
initial_velocities = velocities.copy()
initial_masses = masses.copy()
initial_body_visual_scales = body_visual_scales.copy()

# Calculate physical body radii from mass and density.
SOLAR_MASS_KG = 1.98847e30
AU_METERS = 1.495978707e11
MIN_MASS_SOLAR = 1e-12
MIN_VOLUME_KM3 = 1e-12
MIN_VISUAL_SCALE = 0.01
MIN_ROTATION_PERIOD_DAYS = 0.01

planet_radii_au = []

for mass, density in zip(planet_masses, planet_densities):
    mass_kg = mass * SOLAR_MASS_KG
    density_kg_m3 = density * 1000.0
    radius_m = (3.0 * mass_kg / (4.0 * math.pi * density_kg_m3)) ** (1.0 / 3.0)
    planet_radii_au.append(radius_m / AU_METERS)

planet_radii_au = np.array(planet_radii_au)

star_radii_au = []

for mass, density in zip(star_masses, star_densities):
    mass_kg = mass * SOLAR_MASS_KG
    density_kg_m3 = density * 1000.0

    radius_m = (3.0 * mass_kg / (4.0 * math.pi * density_kg_m3)) ** (1.0 / 3.0)

    star_radii_au.append(radius_m / AU_METERS)

star_radii_au = np.array(star_radii_au)

star_volumes_km3 = (4.0 / 3.0) * math.pi * (star_radii_au * AU_METERS / 1000.0) ** 3
initial_star_volumes_km3 = star_volumes_km3.copy()

collision_radii = np.concatenate([star_radii_au, planet_radii_au])
REFERENCE_CAMERA_DISTANCE_AU = 4.5
REFERENCE_PROJECTION_SCALE = 1.0 / math.tan(math.radians(60.0) * 0.5)
body_visual_exaggeration = (
    body_reference_pixel_sizes
    * REFERENCE_CAMERA_DISTANCE_AU
    / (REFERENCE_PROJECTION_SCALE * HEIGHT * collision_radii)
)
# Planet volumes in km cube
planet_volumes_km3 = (4.0 / 3.0) * math.pi * (planet_radii_au * AU_METERS / 1000.0) ** 3
initial_planet_volumes_km3 = planet_volumes_km3.copy()


def update_star_physics(body_index):
    volume_m3 = star_volumes_km3[body_index] * 1e9
    mass_kg = masses[body_index] * SOLAR_MASS_KG
    density_kg_m3 = mass_kg / volume_m3
    star_densities[body_index] = density_kg_m3 / 1000.0

    radius_km = (3.0 * star_volumes_km3[body_index] / (4.0 * math.pi)) ** (1.0 / 3.0)
    radius_au = radius_km * 1000.0 / AU_METERS
    star_radii_au[body_index] = radius_au
    collision_radii[body_index] = radius_au


def update_planet_physics(body_index):
    planet_index = body_index - NUM_STARS

    volume_m3 = planet_volumes_km3[planet_index] * 1e9
    mass_kg = masses[body_index] * SOLAR_MASS_KG
    density_kg_m3 = mass_kg / volume_m3
    planet_densities[planet_index] = density_kg_m3 / 1000.0

    radius_km = (3.0 * planet_volumes_km3[planet_index] / (4.0 * math.pi)) ** (1.0 / 3.0)

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

    point_mass_factor = G * masses[None, :] / safe_distance**3
    inside_sphere_factor = G * masses[None, :] / source_radii**3

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

    previous_positions = positions.copy()
    previous_velocities = velocities.copy()

    a0 = acceleration(positions)
    velocities += 0.5 * a0 * DT
    positions += velocities * DT

    collision = detect_collision(positions, previous_positions)

    if collision is not None:
        body_a, body_b = collision
        positions[:] = previous_positions
        velocities[:] = previous_velocities
        print(
            f"COLLISION! Simulation stopped before overlap between Body {body_a} and Body {body_b}."
        )
        paused = True
        return False

    a1 = acceleration(positions)
    velocities += 0.5 * a1 * DT
    return True


# collision detection
def detect_collision(pos, previous_pos=None):
    if previous_pos is None:
        pairwise_offsets = pos[:, None, :] - pos[None, :, :]
        pairwise_distances = np.linalg.norm(pairwise_offsets, axis=2)
    else:
        pairwise_offsets = previous_pos[:, None, :] - previous_pos[None, :, :]
        pairwise_motion = (pos - previous_pos)[:, None, :] - (pos - previous_pos)[None, :, :]
        motion_squared = np.einsum(
            "ijk,ijk->ij",
            pairwise_motion,
            pairwise_motion,
        )
        start_motion_dot = np.einsum(
            "ijk,ijk->ij",
            pairwise_offsets,
            pairwise_motion,
        )
        closest_time = np.zeros_like(motion_squared)
        np.divide(
            -start_motion_dot,
            motion_squared,
            out=closest_time,
            where=motion_squared > 1e-20,
        )
        np.clip(closest_time, 0.0, 1.0, out=closest_time)
        closest_offsets = pairwise_offsets + closest_time[:, :, None] * pairwise_motion
        pairwise_distances = np.linalg.norm(closest_offsets, axis=2)

    minimum_distances = collision_radii[:, None] + collision_radii[None, :]
    colliding_pairs = np.argwhere(np.triu(pairwise_distances <= minimum_distances, k=1))

    if len(colliding_pairs) == 0:
        return None

    return tuple(int(index) for index in colliding_pairs[0])


# clipping fix
def limit_visual_overlap(centers, radii, floor_radii, gap=0.9):
    """Shrink displayed radii so no two spheres overlap.
    Never goes below the true physical radius."""
    delta = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, np.inf)

    radius_sum = np.maximum(radii[:, None] + radii[None, :], 1e-30)
    pair_scale = np.minimum(1.0, gap * dist / radius_sum)
    body_scale = pair_scale.min(axis=1)

    return np.maximum(radii * body_scale, floor_radii)


# body controls init for gui


def set_body_value(sender, app_data, user_data):
    property_name, body_index, component = user_data

    if property_name == "position":
        positions[body_index, component] = app_data
        history[body_index].clear()
        history[body_index].append((simulation_years, positions[body_index].copy()))

    elif property_name == "velocity":
        velocities[body_index, component] = app_data

    elif property_name == "mass":
        if not math.isfinite(app_data) or app_data < MIN_MASS_SOLAR:
            print(f"Mass must be finite and at least {MIN_MASS_SOLAR:g} solar masses.")
            dpg.set_value(sender, float(masses[body_index]))
            return

        masses[body_index] = app_data

        if body_index >= NUM_STARS:
            update_planet_physics(body_index)
        else:
            update_star_physics(body_index)

        density = (
            star_densities[body_index]
            if body_index < NUM_STARS
            else planet_densities[body_index - NUM_STARS]
        )
        dpg.set_value(
            f"density_{body_index}",
            f"Density: {density:.3f} g/cm³",
        )

        for i in range(NUM_BODIES):
            history[i].clear()
            history[i].append((simulation_years, positions[i].copy()))

    elif property_name == "volume":
        if not math.isfinite(app_data) or app_data < MIN_VOLUME_KM3:
            print(f"Volume must be finite and at least {MIN_VOLUME_KM3:g} km³.")
            dpg.set_value(
                sender,
                float(
                    planet_volumes_km3[body_index - NUM_STARS]
                    if body_index >= NUM_STARS
                    else star_volumes_km3[body_index]
                ),
            )
            return

        if body_index >= NUM_STARS:
            planet_index = body_index - NUM_STARS
            planet_volumes_km3[planet_index] = app_data
            update_planet_physics(body_index)
            density = planet_densities[planet_index]
        else:
            star_volumes_km3[body_index] = app_data
            update_star_physics(body_index)
            density = star_densities[body_index]
        dpg.set_value(
            f"density_{body_index}",
            f"Density: {density:.3f} g/cm³",
        )

    elif property_name == "size":
        if not math.isfinite(app_data) or app_data < MIN_VISUAL_SCALE:
            print(f"Visual radius multiplier must be finite and at least {MIN_VISUAL_SCALE:g}.")
            dpg.set_value(sender, float(body_visual_scales[body_index]))
            return

        body_visual_scales[body_index] = app_data

    elif property_name == "rotation_period":
        if not math.isfinite(app_data) or app_data < MIN_ROTATION_PERIOD_DAYS:
            print(f"Rotation period must be finite and at least {MIN_ROTATION_PERIOD_DAYS:g} days.")
            dpg.set_value(
                sender,
                float(rotation_period_years[body_index] * 365.25),
            )
            return

        old_period_years = rotation_period_years[body_index]
        new_period_years = app_data / 365.25
        current_angle = (
            rotation_phases[body_index] + 2.0 * math.pi * simulation_years / old_period_years
        )
        rotation_period_years[body_index] = new_period_years
        rotation_phases[body_index] = (
            current_angle - 2.0 * math.pi * simulation_years / new_period_years
        ) % (2.0 * math.pi)


# reset
def reset_simulation():
    global simulation_years, paused, last_trail_update

    positions[:] = initial_positions
    velocities[:] = initial_velocities
    masses[:] = initial_masses
    body_visual_scales[:] = initial_body_visual_scales
    rotation_period_years[:] = initial_rotation_period_years
    rotation_phases[:] = initial_rotation_phases
    star_volumes_km3[:] = initial_star_volumes_km3
    planet_volumes_km3[:] = initial_planet_volumes_km3

    for body_index in range(NUM_BODIES):
        if body_index < NUM_STARS:
            update_star_physics(body_index)
        else:
            update_planet_physics(body_index)

    simulation_years = 0.0
    paused = False

    for i in range(NUM_BODIES):
        history[i].clear()
        start_position = positions[i].copy()
        history[i].append((simulation_years, start_position.copy()))
        history[i].append((simulation_years + DT, start_position.copy()))

        for component in range(3):
            dpg.set_value(f"position_{i}_{component}", float(positions[i, component]))
            dpg.set_value(f"velocity_{i}_{component}", float(velocities[i, component]))

        dpg.set_value(f"mass_{i}", float(masses[i]))
        dpg.set_value(f"size_{i}", float(body_visual_scales[i]))
        dpg.set_value(
            f"rotation_period_{i}",
            float(rotation_period_years[i] * 365.25),
        )

        if i < NUM_STARS:
            volume = star_volumes_km3[i]
            density = star_densities[i]
        else:
            planet_index = i - NUM_STARS
            volume = planet_volumes_km3[planet_index]
            density = planet_densities[planet_index]
        dpg.set_value(f"volume_{i}", float(volume))
        dpg.set_value(
            f"density_{i}",
            f"Density: {density:.3f} g/cm³",
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
                width=200,
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
                width=200,
            )

        dpg.add_separator()

        mass = float(masses[body_index])
        mass_step = max(mass * 0.01, MIN_MASS_SOLAR)
        dpg.add_input_float(
            tag=f"mass_{body_index}",
            label="Mass (solar masses)",
            default_value=mass,
            format="%.3e",
            step=mass_step,
            step_fast=mass_step * 10.0,
            min_value=MIN_MASS_SOLAR,
            min_clamped=True,
            callback=set_body_value,
            user_data=("mass", body_index, None),
            width=200,
        )

        if body_index < NUM_STARS:
            volume = star_volumes_km3[body_index]
            density = star_densities[body_index]
        else:
            planet_index = body_index - NUM_STARS
            volume = planet_volumes_km3[planet_index]
            density = planet_densities[planet_index]
        dpg.add_input_float(
            tag=f"volume_{body_index}",
            label="Volume (km³)",
            default_value=float(volume),
            format="%.3e",
            step=max(float(volume) * 0.01, MIN_VOLUME_KM3),
            step_fast=max(float(volume) * 0.1, MIN_VOLUME_KM3),
            min_value=MIN_VOLUME_KM3,
            min_clamped=True,
            callback=set_body_value,
            user_data=("volume", body_index, None),
            width=200,
        )
        dpg.add_text(f"Density: {density:.3f} g/cm³", tag=f"density_{body_index}")

        dpg.add_input_float(
            tag=f"size_{body_index}",
            label="Visual Radius Multiplier",
            default_value=float(body_visual_scales[body_index]),
            format="%.2f",
            step=0.1,
            step_fast=0.5,
            min_value=MIN_VISUAL_SCALE,
            min_clamped=True,
            callback=set_body_value,
            user_data=("size", body_index, None),
            width=200,
        )
        period_days = float(rotation_period_years[body_index] * 365.25)
        dpg.add_input_float(
            tag=f"rotation_period_{body_index}",
            label="Rotation Period (days)",
            default_value=period_days,
            format="%.4f",
            step=max(period_days * 0.01, MIN_ROTATION_PERIOD_DAYS * 0.1),
            step_fast=max(period_days * 0.1, MIN_ROTATION_PERIOD_DAYS),
            min_value=MIN_ROTATION_PERIOD_DAYS,
            min_clamped=True,
            callback=set_body_value,
            user_data=("rotation_period", body_index, None),
            width=200,
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
    if target_time <= history[0][0]:
        return history[0][1]

    if target_time >= history[-1][0]:
        return current_position

    low = 0
    high = len(history) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if history[middle][0] <= target_time:
            low = middle
        else:
            high = middle

    time_a, position_a = history[low]
    time_b, position_b = history[high]
    fraction = (target_time - time_a) / (time_b - time_a)
    return position_a + (position_b - position_a) * fraction


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
BACKGROUND_STAR_LOD_COUNT = 24000
BACKGROUND_CHUNK_OFFSETS = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
BACKGROUND_VISIBLE_STAR_COUNT = STARS_PER_CHUNK * len(BACKGROUND_CHUNK_OFFSETS)

# Kept for the existing deterministic nebula generation below.
rng = np.random.default_rng(42)


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

    star_data = np.column_stack(
        [
            star_positions.astype(np.float32),
            colors,
            sizes,
        ]
    ).astype(np.float32)
    chunk_rng.shuffle(star_data, axis=0)
    return star_data


def generate_visible_stars(center_chunk):
    chunk_data = []

    for offset_x, offset_y, offset_z in BACKGROUND_CHUNK_OFFSETS:
        chunk_data.append(
            generate_star_chunk(
                center_chunk[0] + offset_x,
                center_chunk[1] + offset_y,
                center_chunk[2] + offset_z,
            )
        )

    return np.vstack(chunk_data).astype(np.float32)


# ======================================================
# nebula gen
# ======================================================
NUM_NEBULAE = 32

nebula_directions = rng.normal(size=(NUM_NEBULAE, 3))
nebula_directions /= np.linalg.norm(nebula_directions, axis=1, keepdims=True)

nebula_distances = rng.uniform(12.0, 85.0, size=NUM_NEBULAE)
nebula_positions = nebula_directions * nebula_distances[:, None]
nebula_radii = rng.uniform(2.0, 5.0, size=NUM_NEBULAE)

nebula_palette = np.array(
    [
        (0.035, 0.012, 0.075),  # violet
        (0.008, 0.045, 0.055),  # teal
        (0.055, 0.014, 0.025),  # rose
        (0.050, 0.032, 0.008),  # amber
    ],
    dtype=np.float32,
)

nebula_colors = nebula_palette[rng.integers(0, len(nebula_palette), size=NUM_NEBULAE)]

nebula_strengths = rng.uniform(0.35, 0.8, size=NUM_NEBULAE).astype(np.float32)

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

        if abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0 or not -1.0 <= ndc[2] <= 1.0:
            continue

        texture, (label_width, label_height) = system_label_textures[system_index]

        # Place the label just above the projected system center.
        label_center = np.array(
            [
                ndc[0],
                ndc[1] + (label_height * 0.5 + 12.0) * 2.0 / height,
            ]
        )

        label_size = np.array(
            [
                label_width * 2.0 / width,
                label_height * 2.0 / height,
            ]
        )

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

glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.SAMPLES, 8)
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

window = glfw.create_window(WIDTH, HEIGHT, "Three Stars + Ten Planets", None, None)

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
nebula_program["nebula_centers"].write(nebula_centers_data.tobytes())
nebula_program["nebula_colors"].write(nebula_colors_data.tobytes())
nebula_vao = ctx.vertex_array(nebula_program, [])
label_program = ctx.program(
    vertex_shader=LABEL_VERTEX_SHADER,
    fragment_shader=LABEL_FRAGMENT_SHADER,
)
label_program["label_texture"].value = 0
label_vao = ctx.vertex_array(label_program, [])


def load_label_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a system font on Windows/macOS/Linux"""
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


label_font = load_label_font(18)

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
            chunk_key for chunk_key in wanted if chunk_key not in background_chunks
        ]

        pending_background_chunks.sort(
            key=lambda key: sum((key[axis] - center_chunk[axis]) ** 2 for axis in range(3))
        )
        loaded_background_chunk = center_chunk

    if pending_background_chunks:
        chunk_key = pending_background_chunks.pop(0)
        data = generate_star_chunk(*chunk_key)
        buffer = ctx.buffer(data.tobytes())
        vao = ctx.vertex_array(
            point_program,
            [(buffer, "3f 3f 1f", "in_position", "in_color", "in_size")],
        )
        background_chunks[chunk_key] = (buffer, vao)


def chunk_is_visible(chunk_key, forward, right, camera_up, aspect, far_plane):
    chunk_center = (np.asarray(chunk_key, dtype=np.float64) + 0.5) * BACKGROUND_CHUNK_SIZE
    relative = chunk_center - camera_position

    depth = float(np.dot(relative, forward))
    half_size = BACKGROUND_CHUNK_SIZE * 0.5
    depth_extent = half_size * np.abs(forward).sum()
    horizontal_extent = half_size * np.abs(right).sum()
    vertical_extent = half_size * np.abs(camera_up).sum()

    if depth + depth_extent < 0.1 or depth - depth_extent > far_plane:
        return False

    tan_half_vertical = math.tan(math.radians(60.0) * 0.5)
    farthest_depth = max(depth + depth_extent, 0.0)
    half_height = farthest_depth * tan_half_vertical
    half_width = half_height * aspect

    horizontal = abs(float(np.dot(relative, right)))
    vertical = abs(float(np.dot(relative, camera_up)))

    return (
        horizontal <= half_width + horizontal_extent and vertical <= half_height + vertical_extent
    )


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


sphere_vertices, sphere_indices = create_sphere_mesh()
sphere_buffer = ctx.buffer(sphere_vertices.tobytes())
sphere_index_buffer = ctx.buffer(sphere_indices.tobytes())
body_buffer = ctx.buffer(reserve=NUM_BODIES * 16 * 4)  # was 13 floats per body
star_light_data = np.zeros((MAX_STAR_LIGHTS, 4), dtype=np.float32)
star_light_color_data = np.zeros((MAX_STAR_LIGHTS, 4), dtype=np.float32)
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
MAX_TRAIL_VERTICES = 2 * (TRAIL_LENGTH + HISTORY_TRIM_BATCH) + 1
last_trail_update = 0.0


def smooth_trail_points(points, samples_per_segment=2):
    if len(points) < 3:
        return points

    p0 = np.vstack([points[0], points[:-2]])
    p1 = points[:-1]
    p2 = points[1:]
    p3 = np.vstack([points[2:], points[-1]])
    t = np.linspace(
        0.0,
        1.0,
        samples_per_segment,
        endpoint=False,
        dtype=np.float32,
    )[None, :, None]

    segments = 0.5 * (
        2.0 * p1[:, None, :]
        + (-p0 + p2)[:, None, :] * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[:, None, :] * t**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[:, None, :] * t**3
    )
    return np.vstack(
        [
            segments.reshape(-1, 3),
            points[-1],
        ]
    )


def trim_trail_at_body(points, center, body_radius):
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
    discriminant = max(
        quadratic_b**2 - 4.0 * quadratic_a * quadratic_c,
        0.0,
    )
    intersection_fraction = (-quadratic_b - math.sqrt(discriminant)) / (2.0 * quadratic_a)
    intersection_fraction = float(np.clip(intersection_fraction, 0.0, 1.0))
    intersection = segment_start + intersection_fraction * segment
    intersection = center + (intersection - center) * 1.01

    return np.vstack(
        [
            points[: last_outside + 1],
            intersection.astype(np.float32),
        ]
    )


def update_trails(displayed_positions, displayed_radii):
    for i in range(NUM_BODIES):
        hist = history[i]
        if len(hist) < 2:
            trail_vertex_counts[i] = 0
            continue

        delay = float(np.linalg.norm(positions[i] - OBSERVER) / C_AU_PER_YEAR)
        target_time = simulation_years - delay
        times = np.fromiter(
            (entry[0] for entry in hist),
            dtype=np.float64,
            count=len(hist),
        )
        cut = int(np.searchsorted(times, target_time, side="right"))
        if cut < 1:
            trail_vertex_counts[i] = 0
            continue

        points = np.array(
            [entry[1] for entry in hist[:cut]],
            dtype=np.float32,
        ).reshape(-1, 3)
        points = np.vstack(
            [
                points,
                displayed_positions[i].astype(np.float32),
            ]
        )
        points = smooth_trail_points(points)
        points = trim_trail_at_body(
            points,
            displayed_positions[i],
            float(displayed_radii[i]),
        )
        if len(points) < 2:
            trail_vertex_counts[i] = 0
            continue

        data = np.empty((len(points), 6), dtype=np.float32)
        data[:, :3] = points
        data[:, 3:] = body_colors[i]

        if trail_buffers[i] is None:
            trail_buffers[i] = ctx.buffer(
                reserve=MAX_TRAIL_VERTICES * 6 * 4,
                dynamic=True,
            )
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
camera_body_positions = positions.copy()
camera_body_radii = collision_radii.copy()
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
    return np.array(
        [
            math.cos(camera_yaw) * math.cos(camera_pitch),
            math.sin(camera_pitch),
            math.sin(camera_yaw) * math.cos(camera_pitch),
        ],
        dtype=np.float64,
    )


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

        camera_pitch = float(np.clip(camera_pitch, -1.5, 1.5))

    last_mouse_x = x
    last_mouse_y = y


def scroll_callback(window, x_offset, y_offset):
    global camera_position

    constrain_camera_to_bodies(camera_body_positions, camera_body_radii)
    start = camera_position.copy()
    forward = get_camera_forward()
    candidate = start + forward * y_offset * 0.5
    camera_position = resolve_camera_motion(
        start,
        candidate,
        camera_body_positions,
        camera_body_radii,
    )


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

    constrain_camera_to_bodies(camera_body_positions, camera_body_radii)
    start = camera_position.copy()
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
        candidate = start + movement * camera_speed * delta_time
        camera_position = resolve_camera_motion(
            start,
            candidate,
            camera_body_positions,
            camera_body_radii,
        )


def constrain_camera_to_bodies(centers, radii):
    global camera_position

    for body_index, (center, radius) in enumerate(zip(centers, radii)):
        if body_index >= NUM_STARS and not show_planets:
            continue

        offset = camera_position - center
        distance = float(np.linalg.norm(offset))
        minimum_distance = radius + max(radius * 0.05, 1e-3)
        if distance >= minimum_distance:
            continue

        if distance < 1e-12:
            offset = -get_camera_forward()
        else:
            offset /= distance
        camera_position = center + offset * minimum_distance


def resolve_camera_motion(start, end, centers, radii):
    displacement = end - start
    displacement_squared = float(np.dot(displacement, displacement))
    if displacement_squared < 1e-20:
        return end

    earliest_intersection = 1.0
    displacement_length = math.sqrt(displacement_squared)
    for body_index, (center, radius) in enumerate(zip(centers, radii)):
        if body_index >= NUM_STARS and not show_planets:
            continue

        safe_radius = radius + max(radius * 0.05, 1e-3)
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
        safe_fraction = max(
            0.0,
            earliest_intersection - 1e-3 / displacement_length,
        )
        return start + displacement * safe_fraction
    return end


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

    with dpg.theme() as hud_theme, dpg.theme_component(dpg.mvAll):
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
            dpg.set_item_label("pause_button", "Resume" if paused else "Pause")

        dpg.add_button(label="Reset Simulation", callback=reset_simulation)

        dpg.add_button(label="Pause", tag="pause_button", callback=toggle_pause)

        dpg.add_slider_float(
            label="Time Speed",
            min_value=0.1,
            max_value=10.0,
            default_value=1.0,
            callback=lambda s, a: set_time_scale(a),
        )

        dpg.add_checkbox(
            label="Show Trajectories",
            default_value=True,
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
            callback=lambda s, a: set_star_scale(a),
        )

        dpg.add_slider_float(
            label="Planet Size Exaggeration",
            min_value=0.25,
            max_value=4.0,
            default_value=planet_scale,
            callback=lambda s, a: set_planet_scale(a),
        )
        dpg.add_text("Display exaggeration is visual only; physics uses physical mass and volume.")

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
    dpg.create_viewport(title="Simulation Controls", width=470, height=850, vsync=False)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    glfw.make_context_current(window)


def render_control_panel_frame():
    """Draw control-panel frame"""
    global control_panel_open

    if not control_panel_open:
        return
    if not dpg.is_dearpygui_running():
        control_panel_open = False
        return

    dpg.render_dearpygui_frame()
    glfw.make_context_current(window)


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


# ============================================================
# control panel (macOS forbids NSWindow on not main threads)
# ============================================================
run_control_panel()
control_panel_open = True
# frame pass
last_frame_time = glfw.get_time()
# ============================================================
# MAIN LOOP
# ============================================================
try:
    while not glfw.window_should_close(window):
        glfw.poll_events()
        render_control_panel_frame()
        current_time = glfw.get_time()
        delta_time = max(current_time - last_frame_time, 1e-6)
        last_frame_time = current_time

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        while not travel_requests.empty():
            travel_to_system(travel_requests.get())

        # ==========================================
        # ADVANCE ALL 13 BODIES
        # ==========================================

        steps = max(1, int(STEPS_PER_FRAME * time_scale))
        if not paused:
            for _ in range(steps):
                # Advance physics by one timestep
                if not physics_step():
                    break

                simulation_years += DT

                # Save trajectory history
                for i in range(NUM_BODIES):
                    history[i].append((simulation_years, positions[i].copy()))

                    if len(history[i]) > TRAIL_LENGTH + HISTORY_TRIM_BATCH:
                        del history[i][:HISTORY_TRIM_BATCH]

        # ==========================================#
        # REFRESH TRAJECTORY BUFFERS
        # ==========================================#

        # ==========================================
        # LIGHT-TRAVEL-TIME POSITIONS
        # ==========================================

        displayed_positions = positions.copy()

        for i in range(NUM_BODIES):
            displayed_positions[i] = apparent_position(i, simulation_years, history)

        displayed_radii = collision_radii * body_visual_exaggeration * body_visual_scales
        displayed_radii[:NUM_STARS] *= star_scale
        displayed_radii[NUM_STARS:] *= planet_scale

        n_visible = NUM_BODIES if show_planets else NUM_STARS
        displayed_radii[:n_visible] = limit_visual_overlap(
            displayed_positions[:n_visible],
            displayed_radii[:n_visible],
            collision_radii[:n_visible],
        )

        camera_body_positions[:] = displayed_positions
        camera_body_radii[:] = displayed_radii
        now = time.time()
        trail_update_interval = TRAIL_UPDATE_INTERVAL / max(time_scale, 1.0)
        if now - last_trail_update >= trail_update_interval:
            simulation_interval = (
                DT * steps * trail_update_interval / delta_time if not paused else 0.0
            )
            trail_cut_radii = (
                displayed_radii + np.linalg.norm(velocities, axis=1) * simulation_interval
            )
            update_trails(displayed_positions, trail_cut_radii)
            last_trail_update = now

        rotation_angles = (
            rotation_phases + 2.0 * math.pi * simulation_years / rotation_period_years
        ) % (2.0 * math.pi)

        star_light_data.fill(0.0)
        star_light_color_data.fill(0.0)
        star_light_data[:NUM_STARS, :3] = displayed_positions[:NUM_STARS]
        star_light_data[:NUM_STARS, 3] = np.maximum(masses[:NUM_STARS], MIN_MASS_SOLAR) ** 3.5
        star_light_color_data[:NUM_STARS, :3] = np.asarray(
            body_colors[:NUM_STARS],
            dtype=np.float32,
        )

        body_data = np.hstack(
            [
                displayed_positions.astype(np.float32),
                np.array(body_colors, dtype=np.float32),
                displayed_radii[:, None].astype(np.float32),
                body_is_star[:, None],
                rotation_axes.astype(np.float32),
                rotation_angles[:, None].astype(np.float32),
                body_is_gas[:, None],
                body_seeds[:, None],
                body_oblateness[:, None],
                body_roughness[:, None],
            ]
        ).astype(np.float32)

        body_buffer.write(body_data.tobytes())

        update_camera(delta_time)
        constrain_camera_to_bodies(displayed_positions, displayed_radii)

        # CAMERA

        forward = get_camera_forward()

        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        width, height = glfw.get_framebuffer_size(window)
        width = max(width, 1)
        height = max(height, 1)

        far_plane = max(
            200.0,
            float(
                np.max(
                    np.linalg.norm(
                        displayed_positions - camera_position,
                        axis=1,
                    )
                    + displayed_radii
                )
            )
            + 100.0,
        )
        visible_body_indices = slice(None) if show_planets else slice(0, NUM_STARS)
        surface_clearances = (
            np.linalg.norm(
                displayed_positions[visible_body_indices] - camera_position,
                axis=1,
            )
            - displayed_radii[visible_body_indices]
        )
        nearest_surface = max(float(np.min(surface_clearances)), 1e-5)
        near_plane = max(1e-5, min(0.01, nearest_surface * 0.25))

        projection = perspective(
            60.0,
            width / height,
            near_plane,
            far_plane,
        )

        view = look_at(camera_position, camera_position + forward, up)

        mvp = projection @ view

        body_program["view"].write(view.T.astype("f4").tobytes())
        body_program["projection"].write(projection.T.astype("f4").tobytes())
        body_program["star_count"].value = NUM_STARS
        body_program["star_lights"].write(star_light_data.tobytes())
        body_program["star_light_colors"].write(star_light_color_data.tobytes())
        body_program["camera_position"].value = tuple(float(v) for v in camera_position)
        body_program["time"].value = float(current_time)

        point_program["mvp"].write(mvp.T.astype("f4").tobytes())
        line_program["mvp"].write(mvp.T.astype("f4").tobytes())

        ctx.viewport = (0, 0, width, height)
        ctx.clear(0.005, 0.007, 0.02, 1.0)
        nebula_program["resolution"].value = (float(width), float(height))
        nebula_program["inverse_projection"].write(
            np.linalg.inv(projection).T.astype("f4").tobytes()
        )
        nebula_program["inverse_view"].write(np.linalg.inv(view).T.astype("f4").tobytes())
        nebula_program["camera_position"].value = tuple(float(value) for value in camera_position)
        nebula_vao.render(mode=moderngl.TRIANGLES, vertices=3)

        # Background stars
        current_chunk = tuple(np.floor(camera_position / BACKGROUND_CHUNK_SIZE).astype(np.int64))

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
                    vertices=BACKGROUND_STAR_LOD_COUNT,
                )

        # Main bodies
        body_vao.render(
            mode=moderngl.TRIANGLES,
            vertices=len(sphere_indices),  # was len(sphere_vertices)
            instances=NUM_BODIES if show_planets else NUM_STARS,
        )

        # Draw trails against the completed body depth buffer without changing it.
        if show_trails:
            ctx.depth_mask = False
            try:
                for i, vao in enumerate(trail_vaos):
                    if (
                        vao is not None
                        and trail_vertex_counts[i] > 1
                        and (show_planets or i < NUM_STARS)
                    ):
                        vao.render(
                            mode=moderngl.LINE_STRIP,
                            vertices=int(trail_vertex_counts[i]),
                        )
            finally:
                ctx.depth_mask = True

        draw_system_labels(projection, view, width, height)

        glfw.swap_buffers(window)

        glfw.set_window_title(window, f"Three Stars + Ten Planets | {simulation_years:.5f} years")

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
    sphere_index_buffer.release()
    sphere_buffer.release()
    point_program.release()
    line_program.release()
    body_program.release()
    ctx.release()

    dpg.destroy_context()
    glfw.destroy_window(window)
    glfw.terminate()
