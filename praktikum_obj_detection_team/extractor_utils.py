import numpy as np

SCREEN_W = 160.0
SCREEN_H = 210.0


def extract_entity(obj_data, entry):
    """
    Named-tuple entity with .x .y .width .height fields.

    JAX stacks frames on axis-0, most-recent first, so we use index [0]
    to get the current frame — NOT [-1] which is the oldest frame.
    """
    if not (hasattr(obj_data, 'x') and hasattr(obj_data, 'y')):
        return []

    x_off = entry.get("x_offset") or 0
    y_off = entry.get("y_offset") or 0

    w_pad = entry.get("w_padding") or 0
    h_pad = entry.get("h_padding") or 0

    try:
        active = np.atleast_1d(obj_data.active[0]) if hasattr(obj_data, 'active') else None
        xs = np.atleast_1d(obj_data.x[0])
        ys = np.atleast_1d(obj_data.y[0])
        ws = np.atleast_1d(obj_data.width[0])
        hs = np.atleast_1d(obj_data.height[0])
    except (AttributeError, IndexError):
        return []

    boxes = []
    for i in range(len(xs)):
        if active is not None and not bool(active[i]):
            continue

        if xs[i] <= 1 or ys[i] <= 1:
            continue

        x = float(xs[i]) + x_off
        y = float(ys[i]) + y_off
        w = float(ws[i]) + w_pad
        h = float(hs[i]) + h_pad

        if (x == 0 and y == 0) or w == 0 or h == 0:
            continue
        boxes.append((x, y, w, h))
    return boxes


def extract_xy_pairs(obj_data, entry):
    try:
        x_off = entry.get("x_offset") or 0
        y_off = entry.get("y_offset") or 0
        cw = entry.get("obj_w") or entry.get("cell_w") or 8
        ch = entry.get("obj_h") or entry.get("cell_h") or 8
        boxes = []
        for pair in obj_data[0]:
            x = float(pair[0]) + x_off
            y = float(pair[1]) + y_off
            if float(pair[0]) > 1 and float(pair[1]) > 1:
                boxes.append((x, y, cw, ch))
        return boxes
    except Exception:
        return []


def extract_true_2d_grid(obj_data, entry):
    """[4, R, C] — maps tile layouts dynamically based on row/column indices."""
    try:
        grid = obj_data[0]

        if entry.get("transpose_grid"):
            grid = grid.T

        rows, cols = grid.shape
        boxes = []

        cw = entry.get("cell_w") or (SCREEN_W / cols)
        ch = entry.get("cell_h") or (SCREEN_H / rows)

        step_x = entry.get("step_x") or cw
        step_y = entry.get("step_y") or ch

        ox, oy = entry.get("grid_origin_x", 0), entry.get("grid_origin_y", 0)

        av = entry.get("active_value")
        amin = entry.get("active_min")

        for r in range(rows):
            for c in range(cols):
                val = float(grid[r, c])

                is_active = False
                if amin is not None:
                    if val >= float(amin):
                        is_active = True
                elif av is not None:
                    if val == float(av):
                        is_active = True
                else:
                    if val == 1.0:
                        is_active = True

                if is_active:
                    boxes.append((ox + (c * step_x), oy + (r * step_y), cw, ch))
        return boxes
    except Exception:
        return []


def extract_flat_per_row(obj_data, entry):
    """
    [4, 210] — splits continuous vertical contours into manageable chunks
    so extreme bounding box aspect ratios do not break YOLOv8 training.
    """
    x_off = entry.get("x_offset") or 0
    y_off = entry.get("y_offset") or 0
    obj_w = entry.get("obj_w") or 8
    active_v = entry.get("active_value")
    max_run = int(entry.get("max_run_height") or 32)

    try:
        row_vals = np.array(obj_data[0], dtype=float)
    except Exception:
        return []

    if active_v is not None:
        active_mask = row_vals == float(active_v)
    else:
        active_mask = row_vals > 1

    boxes = []

    def _flush_segment(run_start, run_xs):
        for chunk_start in range(0, len(run_xs), max_run):
            chunk = run_xs[chunk_start : chunk_start + max_run]
            x_min = float(min(chunk))
            x_max = float(max(chunk))
            y_top = run_start + chunk_start
            boxes.append((
                x_min + x_off,
                float(y_top) + y_off,
                (x_max - x_min) + float(obj_w),
                float(len(chunk)),
            ))

    in_run = False
    run_start = 0
    run_xs = []

    for y_idx in range(len(row_vals)):
        if bool(active_mask[y_idx]):
            if not in_run:
                in_run = True
                run_start = y_idx
                run_xs = []
            run_xs.append(float(row_vals[y_idx]))
        else:
            if in_run:
                in_run = False
                if run_xs:
                    _flush_segment(run_start, run_xs)
                run_xs = []

    if in_run and run_xs:
        _flush_segment(run_start, run_xs)

    return boxes