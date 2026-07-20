"""
jupyter_utils.py

Small helpers for working safely with ipywidgets and traitlets in notebooks
that students may re-run out of order or multiple times.
"""

import traitlets


def register_click_handler(button, handler):
    """Idempotently attaches `handler` as the click callback for `button`.

    Widgets persist in the kernel between cell re-runs. Calling
    button.on_click(handler) a second time -- for example, because a
    student re-ran the cell that wires up the callback, without re-running
    the cell that created the button -- stacks a second copy of the handler
    onto the button. The next click then fires the callback twice; a third
    re-run makes it three times, and so on, silently.

    This function removes any handler it previously registered on this
    specific button before adding the new one, so re-running the cell
    always leaves exactly one active handler, regardless of how many times
    it's been run.

    Args:
        button (ipywidgets.Button): the button to attach the handler to
        handler (function): callback function; called with the button as
            its only argument, per the ipywidgets on_click convention

    Returns:
        None
    """
    previous_handler = getattr(button, '_rti_click_handler', None)
    if previous_handler is not None:
        button.on_click(previous_handler, remove=True)

    button.on_click(handler)
    button._rti_click_handler = handler


def register_dlink(source, target, transform=None):
    """Idempotently creates a directional traitlets link between `source`
    and `target`.

    Re-running a cell that calls traitlets.dlink(...) creates a second live
    link between the same source and target every time it's re-run. Each
    additional link fires independently whenever source changes, so the
    target's setter (and transform, if any) runs once per accumulated link
    -- extra work every frame that grows silently the more times the cell
    is re-run.

    This function stores the link object it creates on the target widget
    and unlinks the previous one (if any) before creating a new one, so
    re-running the cell always leaves exactly one active link.

    Args:
        source (tuple): (object, trait_name), same as traitlets.dlink
        target (tuple): (object, trait_name), same as traitlets.dlink
        transform (callable, optional): applied to source's value before
            assigning it to target, same as traitlets.dlink

    Returns:
        traitlets.dlink: the newly created link object
    """
    target_obj, target_trait = target
    attr_name = f'_rti_dlink_{target_trait}'

    previous_link = getattr(target_obj, attr_name, None)
    if previous_link is not None:
        previous_link.unlink()

    if transform is not None:
        new_link = traitlets.dlink(source, target, transform)
    else:
        new_link = traitlets.dlink(source, target)

    setattr(target_obj, attr_name, new_link)

    return new_link
