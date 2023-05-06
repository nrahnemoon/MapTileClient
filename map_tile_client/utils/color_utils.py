import hashlib


def get_color(name_str, alpha=False):
    """
    Given a string, returns a consistent RGB.
    """
    name_hash = int(hashlib.sha512(name_str.encode('utf-8')).hexdigest(), 16)
    r = (name_hash & 0xFF0000) >> 16
    g = (name_hash & 0x00FF00) >> 8
    b = name_hash & 0x0000FF
    if alpha:
        return r, g, b, 255
    else:
        return r, g, b
