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
