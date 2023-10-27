#version 420 core

layout(std140, set = 0, binding = 0) uniform block {
	uniform mat4 rotation;
};

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 vVaryingColor;

void main()
{
	gl_Position = rotation * in_position;
	gl_Position.z = 0.0; // Hack to avoid negative z clipping, is there a better way to do this?
	vVaryingColor = vec4(in_color.rgba);
}
