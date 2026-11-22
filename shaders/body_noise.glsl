// Shared by body.vert and body.frag; load_shader() inserts it after #version.
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
