import colorsys


def hex_to_rgb(hex_color):
    """Converts a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    """Converts an RGB tuple to a hex color."""
    return "#{:02x}{:02x}{:02x}".format(*rgb_color)


def sat(hex_color, reduction_ratio=0.5):
    """Reduces the saturation of a hex color by a given reduction ratio."""
    rgb_color = hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(*[x / 255.0 for x in rgb_color])
    s *= 1 - reduction_ratio  # Reduce saturation by the reduction ratio
    v = 1 - (1 - v) * (1 - reduction_ratio)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return rgb_to_hex((int(r * 255), int(g * 255), int(b * 255)))
