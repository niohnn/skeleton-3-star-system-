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
