#version 330
// MAX_STAR_LIGHTS is defined by load_shader().

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
