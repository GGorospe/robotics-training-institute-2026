"""
preview_utils.py

Bandwidth-aware live camera preview for the B- and C-series notebooks.

WHY THIS MODULE EXISTS
----------------------
The original preview wiring linked the camera directly to the image widget:

    register_dlink((camera, 'value'), (image_widget, 'value'),
                   transform=bgr8_to_jpeg)

That link is unconditional. Every frame the capture thread produces is
JPEG-encoded and pushed onto the kernel's iopub socket -> Jupyter server ->
websocket -> browser. Nothing in that chain drops frames when the browser
falls behind; it is a reliable, ordered, buffered pipe, so frames QUEUE.
With five robots on one access point the queue grows without bound and the
students see a preview that is tens of seconds behind reality.

This module fixes that at the source. Frames are gated on a clock BEFORE any
encoding happens, so skipped frames cost almost nothing and -- critically --
are never placed on the wire. The queue stays bounded, so the lag stays
bounded.

It also separates two concerns that were previously tangled together:

    PREVIEW  -- small, low quality, crosses the network many times a second.
                Only needs to be good enough to aim the robot at a block.
    DATASET  -- full resolution, high quality, written to local disk and
                never crosses the network at all.

Because the dataset image is encoded straight from `camera.value` rather than
read back out of the preview widget, shrinking the preview costs nothing in
training-data quality.

SCOPE NOTE: this module deliberately does not depend on jupyter_utils'
register_observer(). It keeps its own small registry so it can be dropped onto
the fleet as a self-contained file. Worth consolidating with jupyter_utils
after the workshop.
"""

import threading
import time

import cv2

# Maps (id(camera), id(widget)) -> the observer callback currently installed,
# so that re-running a notebook cell replaces the preview rather than stacking
# a second one on top of it.
_PREVIEW_REGISTRY = {}


def encode_jpeg(frame, quality=95, max_width=None):
    """Encodes a raw BGR frame (numpy array) to JPEG bytes.

    This is a corrected replacement for jetcam_lite.bgr8_to_jpeg(), which
    accepts a `quality` argument but then calls cv2.imencode() without
    passing it -- so every frame is silently encoded at OpenCV's default
    quality of 95, regardless of what the caller asked for.

    Args:
        frame: BGR numpy array, or None (returns empty bytes).
        quality (int): JPEG quality, 0-100. Actually honored here.
        max_width (int, optional): if given and the frame is wider than
            this, the frame is downscaled so its width equals max_width.
            Aspect ratio is always preserved. This is deliberately a single
            dimension rather than an explicit (width, height): the capture
            pipeline's resolution is set in camera_producer.sh, not here, so
            the preview should adapt to whatever it is handed rather than
            hard-code a shape that could silently start stretching frames.

    Returns:
        bytes: JPEG-encoded image, or empty bytes on failure.
    """
    if frame is None:
        return bytes()

    if max_width is not None and frame.shape[1] > max_width:
        scale = float(max_width) / float(frame.shape[1])
        new_size = (int(round(frame.shape[1] * scale)),
                    int(round(frame.shape[0] * scale)))
        # INTER_AREA is the correct filter for downscaling; it averages
        # source pixels instead of dropping them, so the shrunken preview
        # stays readable rather than turning into aliased noise.
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode('.jpg', frame,
                              [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return bytes()

    return bytes(buffer)


def unregister_preview(camera, widget):
    """Removes a throttled preview previously installed on (camera, widget).

    Safe to call when no preview is installed.

    Args:
        camera: the TraitletCamera instance
        widget: the ipywidgets.Image instance
    """
    key = (id(camera), id(widget))
    callback = _PREVIEW_REGISTRY.pop(key, None)
    if callback is not None:
        try:
            camera.unobserve(callback, names='value')
        except ValueError:
            # Already detached (e.g. by an exclusive re-registration). Fine.
            pass


def register_throttled_preview(camera, widget, fps=5, max_width=320,
                               quality=50, exclusive=True, verbose=True):
    """Streams camera frames to an image widget at a capped frame rate.

    Re-running the cell that calls this is safe: the previous preview is torn
    down before the new one is installed, so handlers never stack.

    Args:
        camera: TraitletCamera instance (must expose a 'value' trait).
        widget: ipywidgets.Image instance to write JPEG bytes into.
        fps (int): maximum frames per second sent to the browser. 4-6 is
            plenty for aiming at a stationary object; the capture thread
            keeps running at full speed underneath, so camera.value is
            always current when a student presses a capture button.
        max_width (int): preview frames are downscaled to this width.
        quality (int): JPEG quality for the preview only.
        exclusive (bool): if True, detach EVERY existing observer on
            camera.value first. This is what removes the old
            register_dlink() wiring when a notebook is patched in place
            without restarting the kernel. Pass False in notebooks where
            something else (e.g. a live inference loop) also observes the
            camera.
        verbose (bool): print a one-line confirmation of the active settings.

    Returns:
        The observer callback, so it can be passed to camera.unobserve()
        directly if needed.
    """
    if fps <= 0:
        raise ValueError('fps must be greater than zero')

    interval = 1.0 / float(fps)

    # Mutable state shared with the closure below. A dict is used rather than
    # `nonlocal` so this stays readable to students who may look inside.
    state = {'last_sent': 0.0, 'widget_sized': False}

    # The observer fires on the camera's capture thread. Only one such thread
    # exists, so this lock is a guard rather than a necessity -- it prevents
    # overlapping encodes if a frame ever arrives while one is still running.
    encode_lock = threading.Lock()

    def _on_new_frame(change):
        now = time.monotonic()

        # THE THROTTLE. Checked before any work is done, so a skipped frame
        # costs one subtraction and one comparison -- and, most importantly,
        # never touches the network.
        if now - state['last_sent'] < interval:
            return

        if not encode_lock.acquire(blocking=False):
            return

        try:
            frame = change['new']
            if frame is None:
                return

            state['last_sent'] = now

            # Size the widget from a real frame rather than trusting
            # camera.width/camera.height. Those defaults (640x360) do match
            # what camera_producer.sh currently emits, but the resize inside
            # TraitletCamera._capture_loop is commented out, so nothing
            # actually enforces the agreement. If the GStreamer caps in
            # camera_producer.sh are ever changed, the attributes would
            # silently disagree with the frames. Measuring costs nothing.
            if not state['widget_sized']:
                widget.width = int(frame.shape[1])
                widget.height = int(frame.shape[0])
                state['widget_sized'] = True

            widget.value = encode_jpeg(frame, quality=quality,
                                       max_width=max_width)
        finally:
            encode_lock.release()

    if exclusive:
        # Detaches the old register_dlink() wiring along with any previous
        # throttled preview. Without this, a student who pulls the patched
        # notebook and re-runs the cell would end up running BOTH the old
        # unthrottled link and the new throttled one.
        camera.unobserve_all('value')
        for existing_key in [k for k in _PREVIEW_REGISTRY if k[0] == id(camera)]:
            _PREVIEW_REGISTRY.pop(existing_key, None)
    else:
        unregister_preview(camera, widget)

    camera.observe(_on_new_frame, names='value')
    _PREVIEW_REGISTRY[(id(camera), id(widget))] = _on_new_frame

    if verbose:
        print(f'Live preview: {fps} fps, {max_width}px wide, quality {quality}.')
        print('Saved images are unaffected -- they stay full resolution.')

    return _on_new_frame
