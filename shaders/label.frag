#version 330

in vec2 uv;
uniform sampler2D label_texture;
out vec4 frag_color;

void main() {
    frag_color = texture(label_texture, uv);
}
