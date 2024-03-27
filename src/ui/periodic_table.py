import logging

import dearpygui.dearpygui as dpg
from attrs import define, field

from history import undoable
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


@define(slots=False, repr=False)
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

        with dpg.window(tag="periodic_table", label="Periodic table", show=False, no_resize=True, height=400):
            with dpg.table(
                width=-1,
                header_row=False,
                no_clip=True,
                scrollX=False,
                scrollY=False,
                precise_widths=True,
                policy=dpg.mvTable_SizingFixedFit,
                borders_outerH=True,
                borders_outerV=True,
                pad_outerX=True,
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
                with dpg.table_row(label="row_separator"):
                    for _ in range(18):
                        dpg.add_group()
                with dpg.table_row(label="row_1"):
                    dpg.add_selectable(
                        label="H ",
                        tag=f"periodic_table_{"H"}_selectable",
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
                        tag=f"periodic_table_{"He"}_selectable",
                        width=20,
                        user_data=2,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_2"):
                    dpg.add_selectable(
                        label="Li",
                        tag=f"periodic_table_{"Li"}_selectable",
                        width=20,
                        user_data=3,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Be",
                        tag=f"periodic_table_{"Be"}_selectable",
                        width=20,
                        user_data=4,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                    for _ in range(10):
                        dpg.add_group()

                    dpg.add_selectable(
                        label="B ",
                        tag=f"periodic_table_{"B"}_selectable",
                        width=20,
                        user_data=5,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="C ",
                        tag=f"periodic_table_{"C"}_selectable",
                        width=20,
                        user_data=6,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="N ",
                        tag=f"periodic_table_{"N"}_selectable",
                        width=20,
                        user_data=7,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="O ",
                        tag=f"periodic_table_{"O"}_selectable",
                        width=20,
                        user_data=8,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="F ",
                        tag=f"periodic_table_{"F"}_selectable",
                        width=20,
                        user_data=9,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ne",
                        tag=f"periodic_table_{"Ne"}_selectable",
                        width=20,
                        user_data=10,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_3"):
                    dpg.add_selectable(
                        label="Na",
                        tag=f"periodic_table_{"Na"}_selectable",
                        width=20,
                        user_data=11,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mg",
                        tag=f"periodic_table_{"Mg"}_selectable",
                        width=20,
                        user_data=12,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                    for _ in range(10):
                        dpg.add_group()

                    dpg.add_selectable(
                        label="Al",
                        tag=f"periodic_table_{"Al"}_selectable",
                        width=20,
                        user_data=13,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Si",
                        tag=f"periodic_table_{"Si"}_selectable",
                        width=20,
                        user_data=14,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="P",
                        tag=f"periodic_table_{"P"}_selectable",
                        width=20,
                        user_data=15,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="S ",
                        tag=f"periodic_table_{"S"}_selectable",
                        width=20,
                        user_data=16,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cl",
                        tag=f"periodic_table_{"Cl"}_selectable",
                        width=20,
                        user_data=17,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ar",
                        tag=f"periodic_table_{"Ar"}_selectable",
                        width=20,
                        user_data=18,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_4"):
                    dpg.add_selectable(
                        label="K ",
                        tag=f"periodic_table_{"K"}_selectable",
                        width=20,
                        user_data=19,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ca",
                        tag=f"periodic_table_{"Ca"}_selectable",
                        width=20,
                        user_data=20,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sc",
                        tag=f"periodic_table_{"Sc"}_selectable",
                        width=20,
                        user_data=21,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ti",
                        tag=f"periodic_table_{"Ti"}_selectable",
                        width=20,
                        user_data=22,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="V ",
                        tag=f"periodic_table_{"V"}_selectable",
                        width=20,
                        user_data=23,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cr",
                        tag=f"periodic_table_{"Cr"}_selectable",
                        width=20,
                        user_data=24,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mn",
                        tag=f"periodic_table_{"Mn"}_selectable",
                        width=20,
                        user_data=25,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Fe",
                        tag=f"periodic_table_{"Fe"}_selectable",
                        width=20,
                        user_data=26,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Co",
                        tag=f"periodic_table_{"Co"}_selectable",
                        width=20,
                        user_data=27,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ni",
                        tag=f"periodic_table_{"Ni"}_selectable",
                        width=20,
                        user_data=28,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cu",
                        tag=f"periodic_table_{"Cu"}_selectable",
                        width=20,
                        user_data=29,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Zn",
                        tag=f"periodic_table_{"Zn"}_selectable",
                        width=20,
                        user_data=30,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ga",
                        tag=f"periodic_table_{"Ga"}_selectable",
                        width=20,
                        user_data=31,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ge",
                        tag=f"periodic_table_{"Ge"}_selectable",
                        width=20,
                        user_data=32,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="As",
                        tag=f"periodic_table_{"As"}_selectable",
                        width=20,
                        user_data=33,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Se",
                        tag=f"periodic_table_{"Se"}_selectable",
                        width=20,
                        user_data=34,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Br",
                        tag=f"periodic_table_{"Br"}_selectable",
                        width=20,
                        user_data=35,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Kr",
                        tag=f"periodic_table_{"Kr"}_selectable",
                        width=20,
                        user_data=36,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_5"):
                    dpg.add_selectable(
                        label="Rb",
                        tag=f"periodic_table_{"Rb"}_selectable",
                        width=20,
                        user_data=37,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sr",
                        tag=f"periodic_table_{"Sr"}_selectable",
                        width=20,
                        user_data=38,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Y ",
                        tag=f"periodic_table_{"Y"}_selectable",
                        width=20,
                        user_data=39,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Zr",
                        tag=f"periodic_table_{"Zr"}_selectable",
                        width=20,
                        user_data=40,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Nb",
                        tag=f"periodic_table_{"Nb"}_selectable",
                        width=20,
                        user_data=41,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mo",
                        tag=f"periodic_table_{"Mo"}_selectable",
                        width=20,
                        user_data=42,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Tc",
                        tag=f"periodic_table_{"Tc"}_selectable",
                        width=20,
                        user_data=43,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ru",
                        tag=f"periodic_table_{"Ru"}_selectable",
                        width=20,
                        user_data=44,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Rh",
                        tag=f"periodic_table_{"Rh"}_selectable",
                        width=20,
                        user_data=45,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pd",
                        tag=f"periodic_table_{"Pd"}_selectable",
                        width=20,
                        user_data=46,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ag",
                        tag=f"periodic_table_{"Ag"}_selectable",
                        width=20,
                        user_data=47,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cd",
                        tag=f"periodic_table_{"Cd"}_selectable",
                        width=20,
                        user_data=48,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="In",
                        tag=f"periodic_table_{"In"}_selectable",
                        width=20,
                        user_data=49,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sn",
                        tag=f"periodic_table_{"Sn"}_selectable",
                        width=20,
                        user_data=50,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sb",
                        tag=f"periodic_table_{"Sb"}_selectable",
                        width=20,
                        user_data=51,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Te",
                        tag=f"periodic_table_{"Te"}_selectable",
                        width=20,
                        user_data=52,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="I",
                        tag=f"periodic_table_{"I"}_selectable",
                        width=20,
                        user_data=53,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Xe",
                        tag=f"periodic_table_{"Xe"}_selectable",
                        width=20,
                        user_data=54,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_6"):
                    dpg.add_selectable(
                        label="Cs",
                        tag=f"periodic_table_{"Cs"}_selectable",
                        width=20,
                        user_data=55,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ba",
                        tag=f"periodic_table_{"Ba"}_selectable",
                        width=20,
                        user_data=56,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="La",
                        tag=f"periodic_table_{"La"}_selectable",
                        width=20,
                        user_data=57,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Hf",
                        tag=f"periodic_table_{"Hf"}_selectable",
                        width=20,
                        user_data=72,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ta",
                        tag=f"periodic_table_{"Ta"}_selectable",
                        width=20,
                        user_data=73,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="W ",
                        tag=f"periodic_table_{"W"}_selectable",
                        width=20,
                        user_data=74,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Re",
                        tag=f"periodic_table_{"Re"}_selectable",
                        width=20,
                        user_data=75,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Os",
                        tag=f"periodic_table_{"Os"}_selectable",
                        width=20,
                        user_data=74,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ir",
                        tag=f"periodic_table_{"Ir"}_selectable",
                        width=20,
                        user_data=77,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pt",
                        tag=f"periodic_table_{"Pt"}_selectable",
                        width=20,
                        user_data=78,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Au",
                        tag=f"periodic_table_{"Au"}_selectable",
                        width=20,
                        user_data=79,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Hg",
                        tag=f"periodic_table_{"Hg"}_selectable",
                        width=20,
                        user_data=80,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Tl",
                        tag=f"periodic_table_{"Tl"}_selectable",
                        width=20,
                        user_data=81,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pb",
                        tag=f"periodic_table_{"Pb"}_selectable",
                        width=20,
                        user_data=82,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Bi",
                        tag=f"periodic_table_{"Bi"}_selectable",
                        width=20,
                        user_data=83,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Po",
                        tag=f"periodic_table_{"Po"}_selectable",
                        width=20,
                        user_data=84,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="At",
                        tag=f"periodic_table_{"At"}_selectable",
                        width=20,
                        user_data=85,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Rn",
                        tag=f"periodic_table_{"Rn"}_selectable",
                        width=20,
                        user_data=86,
                        callback=lambda s, d: self.element_selected(s, d),
                    )

                with dpg.table_row(label="row_7"):
                    dpg.add_selectable(
                        label="Fr",
                        tag=f"periodic_table_{"Fr"}_selectable",
                        width=20,
                        user_data=87,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ra",
                        tag=f"periodic_table_{"Ra"}_selectable",
                        width=20,
                        user_data=88,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ac",
                        tag=f"periodic_table_{"Ac"}_selectable",
                        width=20,
                        user_data=89,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Rf",
                        tag=f"periodic_table_{"Rf"}_selectable",
                        width=20,
                        user_data=104,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Db",
                        tag=f"periodic_table_{"Db"}_selectable",
                        width=20,
                        user_data=105,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sg",
                        tag=f"periodic_table_{"Sg"}_selectable",
                        width=20,
                        user_data=106,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Bh",
                        tag=f"periodic_table_{"Bh"}_selectable",
                        width=20,
                        user_data=107,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Hs",
                        tag=f"periodic_table_{"Hs"}_selectable",
                        width=20,
                        user_data=108,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mt",
                        tag=f"periodic_table_{"Mt"}_selectable",
                        width=20,
                        user_data=109,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ds",
                        tag=f"periodic_table_{"Ds"}_selectable",
                        width=20,
                        user_data=110,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Rg",
                        tag=f"periodic_table_{"Rg"}_selectable",
                        width=20,
                        user_data=111,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cn",
                        tag=f"periodic_table_{"Cn"}_selectable",
                        width=20,
                        user_data=112,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Nh",
                        tag=f"periodic_table_{"Nh"}_selectable",
                        width=20,
                        user_data=113,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Fl",
                        tag=f"periodic_table_{"Fl"}_selectable",
                        width=20,
                        user_data=114,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Mc",
                        tag=f"periodic_table_{"Mc"}_selectable",
                        width=20,
                        user_data=115,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Lv",
                        tag=f"periodic_table_{"Lv"}_selectable",
                        width=20,
                        user_data=116,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ts",
                        tag=f"periodic_table_{"Ts"}_selectable",
                        width=20,
                        user_data=117,
                        enabled=False,
                        callback= lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Og",
                        tag=f"periodic_table_{"Og"}_selectable",
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
                        tag=f"periodic_table_{"Ce"}_selectable",
                        width=20,
                        user_data=58,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pr",
                        tag=f"periodic_table_{"Pr"}_selectable",
                        width=20,
                        user_data=59,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Nd",
                        tag=f"periodic_table_{"Nd"}_selectable",
                        width=20,
                        user_data=60,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pm",
                        tag=f"periodic_table_{"Pm"}_selectable",
                        width=20,
                        user_data=61,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Sm",
                        tag=f"periodic_table_{"Sm"}_selectable",
                        width=20,
                        user_data=62,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Eu",
                        tag=f"periodic_table_{"Eu"}_selectable",
                        width=20,
                        user_data=63,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Gd",
                        tag=f"periodic_table_{"Gd"}_selectable",
                        width=20,
                        user_data=64,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Tb",
                        tag=f"periodic_table_{"Tb"}_selectable",
                        width=20,
                        user_data=65,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Dy",
                        tag=f"periodic_table_{"Dy"}_selectable",
                        width=20,
                        user_data=66,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Ho",
                        tag=f"periodic_table_{"Ho"}_selectable",
                        width=20,
                        user_data=67,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Er",
                        tag=f"periodic_table_{"Er"}_selectable",
                        width=20,
                        user_data=68,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Tm",
                        tag=f"periodic_table_{"Tm"}_selectable",
                        width=20,
                        user_data=69,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Yb",
                        tag=f"periodic_table_{"Yb"}_selectable",
                        width=20,
                        user_data=70,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Lu",
                        tag=f"periodic_table_{"Lu"}_selectable",
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
                        tag=f"periodic_table_{"Th"}_selectable",
                        width=20,
                        user_data=90,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pa",
                        tag=f"periodic_table_{"Pa"}_selectable",
                        width=20,
                        user_data=91,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="U ",
                        tag=f"periodic_table_{"U"}_selectable",
                        width=20,
                        user_data=92,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Np",
                        tag=f"periodic_table_{"Np"}_selectable",
                        width=20,
                        user_data=93,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Pu",
                        tag=f"periodic_table_{"Pu"}_selectable",
                        width=20,
                        user_data=94,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Am",
                        tag=f"periodic_table_{"Am"}_selectable",
                        width=20,
                        user_data=95,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cm",
                        tag=f"periodic_table_{"Cm"}_selectable",
                        width=20,
                        user_data=96,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Bk",
                        tag=f"periodic_table_{"Bk"}_selectable",
                        width=20,
                        user_data=97,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Cf",
                        tag=f"periodic_table_{"Cf"}_selectable",
                        width=20,
                        user_data=98,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Es",
                        tag=f"periodic_table_{"Es"}_selectable",
                        width=20,
                        user_data=99,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Fm",
                        tag=f"periodic_table_{"Fm"}_selectable",
                        width=20,
                        user_data=100,
                        enabled=False,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Md",
                        tag=f"periodic_table_{"Md"}_selectable",
                        width=20,
                        enabled=False,
                        user_data=101,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="No",
                        tag=f"periodic_table_{"No"}_selectable",
                        width=20,
                        enabled=False,
                        user_data=102,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_selectable(
                        label="Lr",
                        tag=f"periodic_table_{"Lr"}_selectable",
                        width=20,
                        enabled=False,
                        user_data=103,
                        callback=lambda s, d: self.element_selected(s, d),
                    )
                    dpg.add_group()

                with dpg.table_row(label="row_separator"):
                    for _ in range(18):
                        dpg.add_group()

        dpg.bind_item_theme("periodic_table", selectable_theme)

    @undoable
    def element_selected(self, sender: DPGItem, state: bool):
        element_number = dpg.get_item_user_data(sender)
        if element_number is None:
            return
        if state:
            self.elements_z_selected.add(element_number)
        else:
            self.elements_z_selected.discard(element_number)

        self.ui_parent.show_emission_plots()

        direct_operation = (self.toggle_element, (element_number, state,), {})
        inverse_operation = (self.toggle_element, (element_number, not state,), {})
        
        yield direct_operation
        yield inverse_operation

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

    def toggle_element(self, element_number: int, state: bool):
        tag = f"periodic_table_{element_z_to_symbol(element_number)}_selectable"
        if state:
            dpg.set_value(tag, True)
            self.elements_z_selected.add(element_number)
        else:
            dpg.set_value(tag, False)
            self.elements_z_selected.discard(element_number)
            
        self.ui_parent.show_emission_plots()
        
