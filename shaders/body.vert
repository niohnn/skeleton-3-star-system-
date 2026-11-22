#version 330

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
