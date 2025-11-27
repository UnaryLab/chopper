from types import MappingProxyType
from typing import Mapping, Tuple


Color = Tuple[float, float, float]
ColorPalette = Mapping[str, Color]


def rgb(r: int, g: int, b: int) -> Color:
    return (r/255, g/255, b/255)


okabe_ito: ColorPalette = MappingProxyType({
    'Black': rgb(0, 0, 0),
    'Green': rgb(0, 158, 115),
    'Blue': rgb(0, 114, 178),
    'Cyan': rgb(86, 180, 233),
    'Yellow': rgb(240, 228, 66),
    'Orange': rgb(230, 159, 0),
    'Red': rgb(213, 94, 0),
    'Pink': rgb(204, 121, 167),
})

catppuccin_latte_colors: ColorPalette = MappingProxyType({
    'Rosewater': rgb(220, 138, 120),
    'Flamingo': rgb(221, 120, 120),
    'Pink': rgb(234, 118, 203),
    'Mauve': rgb(136, 57, 239),
    'Red': rgb(210, 15, 57),
    'Maroon': rgb(230, 69, 83),
    'Peach': rgb(254, 100, 11),
    'Yellow': rgb(223, 142, 29),
    'Green': rgb(64, 160, 43),
    'Teal': rgb(23, 146, 153),
    'Sky': rgb(4, 165, 229),
    'Saphire': rgb(32, 159, 181),
    'Blue': rgb(30, 102, 245),
    'Lavender': rgb(114, 135, 253),
    'Text': rgb(76, 79, 105),
    'Subtext1': rgb(92, 95, 119),
    'Subtext0': rgb(108, 111, 133),
    'Overlay2': rgb(124, 127, 147),
    'Overlay1': rgb(140, 143, 161),
    'Overlay0': rgb(156, 160, 176),
    'Surface2': rgb(172, 176, 190),
    'Surface1': rgb(188, 192, 204),
    'Surface0': rgb(204, 208, 218),
    'Base': rgb(239, 241, 245),
    'Mantle': rgb(230, 233, 239),
    'Crust': rgb(220, 224, 232),
})
catppuccin_mocha_colors: ColorPalette = MappingProxyType({
    'Rosewater': rgb(245, 224, 220),
    'Flamingo': rgb(242, 205, 205),
    'Pink': rgb(245, 194, 231),
    'Mauve': rgb(203, 166, 247),
    'Red': rgb(243, 139, 168),
    'Maroon': rgb(235, 160, 172),
    'Peach': rgb(250, 179, 135),
    'Yellow': rgb(249, 226, 175),
    'Green': rgb(166, 227, 161),
    'Teal': rgb(148, 226, 213),
    'Sky': rgb(137, 220, 235),
    'Saphire': rgb(116, 199, 236),
    'Blue': rgb(137, 180, 250),
    'Lavender': rgb(180, 190, 254),
    'Text': rgb(205, 214, 244),
    'Subtext1': rgb(186, 194, 222),
    'Subtext0': rgb(166, 173, 200),
    'Overlay2': rgb(147, 153, 178),
    'Overlay1': rgb(127, 132, 156),
    'Overlay0': rgb(108, 112, 134),
    'Surface2': rgb(88, 91, 112),
    'Surface1': rgb(69, 71, 90),
    'Surface0': rgb(49, 50, 68),
    'Base': rgb(30, 30, 46),
    'Mantle': rgb(24, 24, 37),
    'Crust': rgb(17, 17, 27),
})
