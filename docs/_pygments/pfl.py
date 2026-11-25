"""Pygments style in the Planet Formation Lab palette, for the eddy docs.

Colours are the PFL tokens from _static/css/eddy.css. Only shades that
reach about 4.5:1 contrast on the paper background are used for text.
"""

from pygments.style import Style
from pygments.token import (Comment, Error, Generic, Keyword, Name, Number,
                            Operator, Punctuation, String, Text, Whitespace)

PAPER = '#FBFAF7'
N300 = '#ABA89B'
N500 = '#585548'
N700 = '#33312A'
N900 = '#1A1916'
DUST_600 = '#C24E00'
GAS_600 = '#136E7A'
MIT_RED = '#A31F34'


class PFLStyle(Style):
    """Warm-neutral code on paper: bold near-black keywords, dust strings."""

    name = 'pfl'
    background_color = PAPER
    highlight_color = '#FFF0E2'  # dust-50, for emphasised lines
    line_number_color = N300

    styles = {
        Text: N700,
        Whitespace: '',
        Error: MIT_RED,

        Comment: 'italic ' + N500,
        Comment.Preproc: 'noitalic ' + N500,

        Keyword: 'bold ' + N900,
        Keyword.Constant: 'bold ' + N900,
        Keyword.Type: GAS_600,

        Operator: N500,
        Operator.Word: 'bold ' + N900,
        Punctuation: N500,

        Name: N700,
        Name.Builtin: GAS_600,
        Name.Builtin.Pseudo: 'italic ' + GAS_600,
        Name.Function: GAS_600,
        Name.Class: GAS_600,
        Name.Namespace: N700,
        Name.Decorator: DUST_600,
        Name.Exception: MIT_RED,

        String: DUST_600,
        String.Doc: 'italic ' + N500,
        String.Escape: 'bold ' + DUST_600,
        String.Interpol: 'bold ' + DUST_600,

        Number: MIT_RED,

        Generic.Prompt: N300,
        Generic.Output: N700,
        Generic.Traceback: MIT_RED,
        Generic.Error: MIT_RED,
        Generic.Heading: 'bold ' + N900,
        Generic.Subheading: 'bold ' + N700,
        Generic.Emph: 'italic',
        Generic.Strong: 'bold',
        Generic.Inserted: GAS_600,
        Generic.Deleted: MIT_RED,
    }
