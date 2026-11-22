#version 330
in vec3 in_position;
in vec3 in_color;
uniform mat4 mvp;
out vec3 vertex_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    vertex_color = in_color;
}
