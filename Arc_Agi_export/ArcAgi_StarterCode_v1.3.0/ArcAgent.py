import numpy as np
from ArcProblem import ArcProblem
from collections import deque


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        train_pairs = []
        for s in arc_problem.training_set():
            inp = s.get_input_data().data()
            out = s.get_output_data().data()
            train_pairs.append((inp, out))

        test_input = arc_problem.test_set().get_input_data().data()
        predictions = []

        # 1) simple direct rules
        simple_rules = [
            lambda x: x.copy(),
            lambda x: np.rot90(x, 1),
            lambda x: np.rot90(x, 2),
            lambda x: np.rot90(x, 3),
            lambda x: np.flipud(x),
            lambda x: np.fliplr(x),
            lambda x: x.T,
            self._crop_nonzero,
            self._crop_and_swap_two_colors,
            self._quad_mirror,
            self._pair_to_line_same_row,
            self._pair_to_line_same_col,
        ]

        for rule in simple_rules:
            if self._fits_all_training(train_pairs, rule):
                self._append_if_new(predictions, rule(test_input))

        # 2) learned boolean panel rules
        for mode in ["and", "or", "xor", "nor"]:
            rule = self._make_panel_rule(train_pairs, mode)
            if rule is not None:
                self._append_if_new(predictions, rule(test_input))

        return predictions[:3]

    # -------------------------
    # utility
    # -------------------------
    def _append_if_new(self, predictions, pred):
        if pred is None:
            return
        for p in predictions:
            if np.array_equal(p, pred):
                return
        predictions.append(pred)

    def _fits_all_training(self, train_pairs, fn):
        for inp, out in train_pairs:
            pred = fn(inp)
            if pred is None or not np.array_equal(pred, out):
                return False
        return True

    # -------------------------
    # basic rules
    # -------------------------
    def _crop_nonzero(self, x):
        coords = np.argwhere(x != 0)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return x[r0:r1 + 1, c0:c1 + 1]

    def _crop_and_swap_two_colors(self, x):
        cropped = self._crop_nonzero(x)
        if cropped is None:
            return None

        colors = np.unique(cropped)
        colors = colors[colors != 0]
        if len(colors) != 2:
            return None

        a, b = colors
        out = cropped.copy()
        out[cropped == a] = b
        out[cropped == b] = a
        return out

    def _quad_mirror(self, x):
        """
        For tasks like 62c24649:
        make a 2x2 symmetric image
        [ x | fliplr(x) ]
        [ flipud(x) | rot180(x) ]
        """
        top = np.hstack([x, np.fliplr(x)])
        bottom = np.hstack([np.flipud(x), np.rot90(x, 2)])
        return np.vstack([top, bottom])

    # -------------------------
    # pair -> line rules
    # -------------------------
    def _pair_to_line_same_row(self, x):
        """
        If a color appears exactly twice in the same row, connect them with a horizontal line.
        Single-pixel objects remain as single pixels.
        Good candidate for 22eb0ac0-like tasks.
        """
        out = np.zeros_like(x)
        colors = [c for c in np.unique(x) if c != 0]

        changed = False
        for color in colors:
            coords = np.argwhere(x == color)
            if len(coords) == 1:
                r, c = coords[0]
                out[r, c] = color
                changed = True
            elif len(coords) == 2:
                (r1, c1), (r2, c2) = coords
                if r1 == r2:
                    cmin, cmax = min(c1, c2), max(c1, c2)
                    out[r1, cmin:cmax + 1] = color
                    changed = True
                elif c1 == c2:
                    # not this rule
                    return None
                else:
                    return None
            else:
                return None

        return out if changed else None

    def _pair_to_line_same_col(self, x):
        """
        Same as above, but vertical.
        """
        out = np.zeros_like(x)
        colors = [c for c in np.unique(x) if c != 0]

        changed = False
        for color in colors:
            coords = np.argwhere(x == color)
            if len(coords) == 1:
                r, c = coords[0]
                out[r, c] = color
                changed = True
            elif len(coords) == 2:
                (r1, c1), (r2, c2) = coords
                if c1 == c2:
                    rmin, rmax = min(r1, r2), max(r1, r2)
                    out[rmin:rmax + 1, c1] = color
                    changed = True
                elif r1 == r2:
                    return None
                else:
                    return None
            else:
                return None

        return out if changed else None

    # -------------------------
    # separator panel rules
    # -------------------------
    def _find_separator(self, x):
        h, w = x.shape

        # uniform nonzero row
        for r in range(h):
            vals = np.unique(x[r, :])
            if len(vals) == 1 and vals[0] != 0:
                a = x[:r, :]
                b = x[r + 1:, :]
                if a.shape == b.shape and a.size > 0:
                    return ("horizontal", r)

        # uniform nonzero col
        for c in range(w):
            vals = np.unique(x[:, c])
            if len(vals) == 1 and vals[0] != 0:
                a = x[:, :c]
                b = x[:, c + 1:]
                if a.shape == b.shape and a.size > 0:
                    return ("vertical", c)

        return None

    def _split_by_separator(self, x):
        sep = self._find_separator(x)
        if sep is None:
            return None

        orientation, idx = sep
        if orientation == "horizontal":
            return x[:idx, :], x[idx + 1:, :]
        return x[:, :idx], x[:, idx + 1:]

    def _bool_mask(self, a, b, mode):
        if mode == "and":
            return (a != 0) & (b != 0)
        if mode == "or":
            return (a != 0) | (b != 0)
        if mode == "xor":
            return ((a != 0) ^ (b != 0))
        if mode == "nor":
            return (a == 0) & (b == 0)
        return None

    def _make_panel_rule(self, train_pairs, mode):
        out_color = None

        for inp, out in train_pairs:
            split = self._split_by_separator(inp)
            if split is None:
                return None

            a, b = split
            if a.shape != out.shape:
                return None

            mask = self._bool_mask(a, b, mode)
            if mask is None:
                return None

            nz = np.unique(out[out != 0])
            if len(nz) > 1:
                return None

            inferred_color = 1 if len(nz) == 0 else int(nz[0])

            candidate = np.zeros_like(a)
            candidate[mask] = inferred_color

            if not np.array_equal(candidate, out):
                return None

            if out_color is None:
                out_color = inferred_color
            elif out_color != inferred_color:
                return None

        if out_color is None:
            return None

        def rule(x):
            split = self._split_by_separator(x)
            if split is None:
                return None
            a, b = split
            mask = self._bool_mask(a, b, mode)
            if mask is None:
                return None
            out = np.zeros_like(a)
            out[mask] = out_color
            return out

        return rule