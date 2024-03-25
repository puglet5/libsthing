import logging

import dearpygui.dearpygui as dpg
from attrs import define, field

from src.utils import DPGItem

logger = logging.getLogger(__name__)

ELEMENT_SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
ELEMENT_NAMES = [
    "Hydrogen",
    "Helium",
    "Lithium",
    "Beryllium",
    "Boron",
    "Carbon",
    "Nitrogen",
    "Oxygen",
    "Fluorine",
    "Neon",
    "Sodium",
    "Magnesium",
    "Aluminium",
    "Silicon",
    "Phosphorus",
    "Sulfur",
    "Chlorine",
    "Argon",
    "Potassium",
    "Calcium",
    "Scandium",
    "Titanium",
    "Vanadium",
    "Chromium",
    "Manganese",
    "Iron",
    "Cobalt",
    "Nickel",
    "Copper",
    "Zinc",
    "Gallium",
    "Germanium",
    "Arsenic",
    "Selenium",
    "Bromine",
    "Krypton",
    "Rubidium",
    "Strontium",
    "Yttrium",
    "Zirconium",
    "Niobium",
    "Molybdenum",
    "Technetium",
    "Ruthenium",
    "Rhodium",
    "Palladium",
    "Silver",
    "Cadmium",
    "Indium",
    "Tin",
    "Antimony",
    "Tellurium",
    "Iodine",
    "Xenon",
    "Caesium",
    "Barium",
    "Lanthanum",
    "Cerium",
    "Praseodymium",
    "Neodymium",
    "Promethium",
    "Samarium",
    "Europium",
    "Gadolinium",
    "Terbium",
    "Dysprosium",
    "Holmium",
    "Erbium",
    "Thulium",
    "Ytterbium",
    "Lutetium",
    "Hafnium",
    "Tantalum",
    "Tungsten",
    "Rhenium",
    "Osmium",
    "Iridium",
    "Platinum",
    "Gold",
    "Mercury",
    "Thallium",
    "Lead",
    "Bismuth",
    "Polonium",
    "Astatine",
    "Radon",
    "Francium",
    "Radium",
    "Actinium",
    "Thorium",
    "Protactinium",
    "Uranium",
    "Neptunium",
    "Plutonium",
    "Americium",
    "Curium",
    "Berkelium",
    "Californium",
    "Einsteinium",
    "Fermium",
    "Mendelevium",
    "Nobelium",
    "Lawrencium",
    "Rutherfordium",
    "Dubnium",
    "Seaborgium",
    "Bohrium",
    "Hassium",
    "Meitnerium",
    "Darmstadtium",
    "Roentgenium",
    "Copernicium",
    "Nihonium",
    "Flerovium",
    "Moscovium",
    "Livermorium",
    "Tennessine",
    "Oganesson",
]


def element_z_to_symbol(z: int) -> str | None:
    if z > 118:
        logger.error("Error: Z out of range")
        return None

    return ELEMENT_SYMBOLS[z - 1]


def element_z_to_name(z) -> str | None:
    if z > 118:
        logger.error("Error: Z out of range")
        return None

    return ELEMENT_NAMES[z - 1]


def element_symbol_to_z(symbol: str):
    try:
        z = ELEMENT_SYMBOLS.index(symbol)
    except ValueError:
        return None

    return z + 1


@define(slots=False)
class PeriodicTable:
    ui_parent: "UI" = field(init=False)  # type:ignore
    elements_z_selected: set[int] = field(init=False, factory=set)

    def __attrs_post_init__(self):
        with dpg.theme() as selectable_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_SelectableTextAlign,
                    0.5,
                    category=dpg.mvThemeCat_Core,
                )

        with dpg.window(
            autosize=True, tag="periodic_table", label="Periodic table", show=False
        ):
            with dpg.table(
                width=-1,
                header_row=False,
                no_clip=True,
                scrollX=False,
                scrollY=False,
                precise_widths=True,
                policy=dpg.mvTable_SizingFixedFit,
            ):
                dpg.add_table_column(
                    no_sort=True,
                    no_header_width=True,
                )
                dpg.add_table_column(
                    no_sort=True,
                    no_header_width=True,
                )
                dpg.add_table_column(no_sort=True, init_width_or_weight=30.0)

                for i in range(15):
                    dpg.add_table_column(
                        no_sort=True,
                        no_header_width=True,
                        no_resize=True,
                    )
                with dpg.table_row(label="row_1"):
                    dpg.add_selectable(
                        label="H ",
                        width=20,
                        user_data=1,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_selectable(
                        label="He",
                        width=20,
                        user_data=2,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_2"):
                    dpg.add_selectable(
                        label="Li",
                        width=20,
                        user_data=3,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Be",
                        width=20,
                        user_data=4,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                    for _ in range(10):
                        dpg.add_group()

                    dpg.add_selectable(
                        label="B ",
                        width=20,
                        user_data=5,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="C ",
                        width=20,
                        user_data=6,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="N ",
                        width=20,
                        user_data=7,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="O ",
                        width=20,
                        user_data=8,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="F ",
                        width=20,
                        user_data=9,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ne",
                        width=20,
                        user_data=10,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_3"):
                    dpg.add_selectable(
                        label="Na",
                        width=20,
                        user_data=11,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mg",
                        width=20,
                        user_data=12,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                    for _ in range(10):
                        dpg.add_group()

                    dpg.add_selectable(
                        label="Al",
                        width=20,
                        user_data=13,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Si",
                        width=20,
                        user_data=14,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="P",
                        width=20,
                        user_data=15,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="S ",
                        width=20,
                        user_data=16,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cl",
                        width=20,
                        user_data=17,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ar",
                        width=20,
                        user_data=18,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_4"):
                    dpg.add_selectable(
                        label="K ",
                        width=20,
                        user_data=19,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ca",
                        width=20,
                        user_data=20,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sc",
                        width=20,
                        user_data=21,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ti",
                        width=20,
                        user_data=22,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="V ",
                        width=20,
                        user_data=23,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cr",
                        width=20,
                        user_data=24,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mn",
                        width=20,
                        user_data=25,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Fe",
                        width=20,
                        user_data=26,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Co",
                        width=20,
                        user_data=27,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ni",
                        width=20,
                        user_data=28,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cu",
                        width=20,
                        user_data=29,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Zn",
                        width=20,
                        user_data=30,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ga",
                        width=20,
                        user_data=31,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ge",
                        width=20,
                        user_data=32,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="As",
                        width=20,
                        user_data=33,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Se",
                        width=20,
                        user_data=34,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Br",
                        width=20,
                        user_data=35,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Kr",
                        width=20,
                        user_data=36,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_5"):
                    dpg.add_selectable(
                        label="Rb",
                        width=20,
                        user_data=37,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sr",
                        width=20,
                        user_data=38,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Y ",
                        width=20,
                        user_data=39,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Zr",
                        width=20,
                        user_data=40,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Nb",
                        width=20,
                        user_data=41,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mo",
                        width=20,
                        user_data=42,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Tc",
                        width=20,
                        user_data=43,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ru",
                        width=20,
                        user_data=44,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Rh",
                        width=20,
                        user_data=45,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pd",
                        width=20,
                        user_data=46,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ag",
                        width=20,
                        user_data=47,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cd",
                        width=20,
                        user_data=48,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="In",
                        width=20,
                        user_data=49,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sn",
                        width=20,
                        user_data=50,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sb",
                        width=20,
                        user_data=51,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Te",
                        width=20,
                        user_data=52,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="I",
                        width=20,
                        user_data=53,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Xe",
                        width=20,
                        user_data=54,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_6"):
                    dpg.add_selectable(
                        label="Cs",
                        width=20,
                        user_data=55,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ba",
                        width=20,
                        user_data=56,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="La",
                        width=20,
                        user_data=57,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Hf",
                        width=20,
                        user_data=72,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ta",
                        width=20,
                        user_data=73,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="W ",
                        width=20,
                        user_data=74,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Re",
                        width=20,
                        user_data=75,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Os",
                        width=20,
                        user_data=74,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ir",
                        width=20,
                        user_data=77,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pt",
                        width=20,
                        user_data=78,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Au",
                        width=20,
                        user_data=79,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Hg",
                        width=20,
                        user_data=80,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Tl",
                        width=20,
                        user_data=81,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pb",
                        width=20,
                        user_data=82,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Bi",
                        width=20,
                        user_data=83,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Po",
                        width=20,
                        user_data=84,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="At",
                        width=20,
                        user_data=85,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Rn",
                        width=20,
                        user_data=86,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_7"):
                    dpg.add_selectable(
                        label="Fr",
                        width=20,
                        user_data=87,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ra",
                        width=20,
                        user_data=88,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ac",
                        width=20,
                        user_data=89,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Rf",
                        width=20,
                        user_data=104,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Db",
                        width=20,
                        user_data=105,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sg",
                        width=20,
                        user_data=106,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Bh",
                        width=20,
                        user_data=107,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Hs",
                        width=20,
                        user_data=108,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mt",
                        width=20,
                        user_data=109,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ds",
                        width=20,
                        user_data=110,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Rg",
                        width=20,
                        user_data=111,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cn",
                        width=20,
                        user_data=112,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Nh",
                        width=20,
                        user_data=113,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Fl",
                        width=20,
                        user_data=114,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mc",
                        width=20,
                        user_data=115,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Lv",
                        width=20,
                        user_data=116,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ts",
                        width=20,
                        user_data=117,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Og",
                        width=20,
                        user_data=118,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_separator"):
                    for _ in range(18):
                        dpg.add_group()
                with dpg.table_row(label="row_separator_extra"):
                    for _ in range(18):
                        dpg.add_group()

                with dpg.table_row(label="row_la"):
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_selectable(
                        label="Ce",
                        width=20,
                        user_data=58,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pr",
                        width=20,
                        user_data=59,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Nd",
                        width=20,
                        user_data=60,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pm",
                        width=20,
                        user_data=61,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sm",
                        width=20,
                        user_data=62,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Eu",
                        width=20,
                        user_data=63,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Gd",
                        width=20,
                        user_data=64,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Tb",
                        width=20,
                        user_data=65,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Dy",
                        width=20,
                        user_data=66,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ho",
                        width=20,
                        user_data=67,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Er",
                        width=20,
                        user_data=68,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Tm",
                        width=20,
                        user_data=69,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Yb",
                        width=20,
                        user_data=70,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Lu",
                        width=20,
                        user_data=71,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_group()

                with dpg.table_row(label="row_ac"):
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_group()
                    dpg.add_selectable(
                        label="Th",
                        width=20,
                        user_data=90,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pa",
                        width=20,
                        user_data=91,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="U ",
                        width=20,
                        user_data=92,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Np",
                        width=20,
                        user_data=93,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pu",
                        width=20,
                        user_data=94,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Am",
                        width=20,
                        user_data=95,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cm",
                        width=20,
                        user_data=96,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Bk",
                        width=20,
                        user_data=97,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cf",
                        width=20,
                        user_data=98,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Es",
                        width=20,
                        user_data=99,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Fm",
                        width=20,
                        user_data=100,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Md",
                        width=20,
                        enabled=False,
                        user_data=101,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="No",
                        width=20,
                        enabled=False,
                        user_data=102,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Lr",
                        enabled=False,
                        width=20,
                        user_data=103,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_group()

        dpg.bind_item_theme("periodic_table", selectable_theme)

    def element_selected(self, sender: DPGItem, state: bool):
        element_number = dpg.get_item_user_data(sender)
        if element_number is None:
            return
        if state:
            self.elements_z_selected.add(element_number)
        else:
            self.elements_z_selected.discard(element_number)

        self.ui_parent.show_emission_plots()

    @property
    def element_symbols_selected(self):
        return {element_z_to_symbol(z) for z in self.elements_z_selected}

    def show(self):
        dpg.show_item("periodic_table")

    def hide(self):
        dpg.hide_item("periodic_table")

    def toggle(self):
        if dpg.is_item_shown("periodic_table"):
            dpg.hide_item("periodic_table")
        else:
            dpg.show_item("periodic_table")

    def clear_selections(self):
        self.elements_z_selected = set()
