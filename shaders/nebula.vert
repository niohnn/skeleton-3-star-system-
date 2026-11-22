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
