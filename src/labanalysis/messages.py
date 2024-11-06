"""
tk common messages module

this module provides an interface to the native message boxes available in Tk 4.2 and newer.

written by Fredrik Lundh, May 1997


options (all have default values):

    - default: which button to make default (one of the reply codes)
    - icon: which icon to display (see below)
    - message: the message to display
    - parent: which window to place the dialog on top of
    - title: dialog title
    - type: dialog type; that is, which buttons to display (see below)
"""

#! IMPORTS


from tkinter.commondialog import Dialog

#! CONSTANTS

# icons
_ERROR = "error"
_INFO = "info"
_QUESTION = "question"
_WARNING = "warning"

# types
_OK = "ok"
_OKCANCEL = "okcancel"
_RETRYCANCEL = "retrycancel"
_YESNO = "yesno"
_YESNOCANCEL = "yesnocancel"

# replies
_RETRY = "retry"
_CANCEL = "cancel"
_YES = "yes"
_NO = "no"


#! CLASS


class _TkMessage(Dialog):
    "A message box"

    command = "tk_messageBox"


#! METHODS


# Rename _icon and _type options to allow overriding them in options
def _show(title=None, message=None, _icon=None, _type=None, **options):
    if _icon and "icon" not in options:
        options["icon"] = _icon
    if _type and "type" not in options:
        options["type"] = _type
    if title:
        options["title"] = title
    if message:
        options["message"] = message
    res = _TkMessage(**options).show()
    # In some Tcl installations, yes/no is converted into a boolean.
    if isinstance(res, bool):
        if res:
            return _YES
        return _NO
    # In others we get a Tcl_Obj.
    return str(res)


def showinfo(title=None, message=None, **options):
    "Show an info message"
    return _show(title, message, _INFO, _OK, **options)


def showwarning(title=None, message=None, **options):
    "Show a warning message"
    return _show(title, message, _WARNING, _OK, **options)


def showerror(title=None, message=None, **options):
    "Show an error message"
    return _show(title, message, _ERROR, _OK, **options)


def askquestion(title=None, message=None, **options):
    "Ask a question"
    return _show(title, message, _QUESTION, _YESNO, **options)


def askokcancel(title=None, message=None, **options):
    "Ask if operation should proceed; return true if the answer is ok"
    s = _show(title, message, _QUESTION, _OKCANCEL, **options)
    return s == _OK


def askyesno(title=None, message=None, **options):
    "Ask a question; return true if the answer is yes"
    s = _show(title, message, _QUESTION, _YESNO, **options)
    return s == _YES


def askyesnocancel(title=None, message=None, **options):
    "Ask a question; return true if the answer is yes, None if cancelled."
    s = _show(title, message, _QUESTION, _YESNOCANCEL, **options)
    # s might be a Tcl index object, so convert it to a string
    s = str(s)
    if s == _CANCEL:
        return None
    return s == _YES


def askretrycancel(title=None, message=None, **options):
    "Ask if operation should be retried; return true if the answer is yes"
    s = _show(title, message, _WARNING, _RETRYCANCEL, **options)
    return s == _RETRY


__all__ = [
    "showinfo",
    "showwarning",
    "showerror",
    "askquestion",
    "askokcancel",
    "askyesno",
    "askyesnocancel",
    "askretrycancel",
]
