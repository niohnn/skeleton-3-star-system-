#version 330
// NUM_NEBULAE is defined by load_shader().

uniform vec2 resolution;
uniform mat4 inverse_projection;
uniform mat4 inverse_view;
uniform vec3 camera_position;
uniform vec4 nebula_centers[NUM_NEBULAE];
uniform vec4 nebula_colors[NUM_NEBULAE];
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

    for (int i = 0; i < NUM_NEBULAE; i++) {
        float density = nebula_blob(
            world_ray,
            nebula_centers[i],
            nebula_colors[i]
        );
        color += nebula_colors[i].rgb * density;
    }

    frag_color = vec4(color, 1.0);
}
