import numpy as np
from ArcProblem import ArcProblem


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

        # Simple fixed rules
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
        ]

        for rule in simple_rules:
            if self._fits_all_training(train_pairs, rule):
                pred = rule(test_input)
                self._append_if_new(predictions, pred)

        # More general panel logic rules
        panel_rules = [
            self._make_panel_rule(train_pairs, mode="and"),
            self._make_panel_rule(train_pairs, mode="nor"),
        ]

        for rule in panel_rules:
            if rule is not None:
                pred = rule(test_input)
                self._append_if_new(predictions, pred)

        return predictions[:3]

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

    def _find_separator(self, x):
        h, w = x.shape

        # Check rows
        for r in range(h):
            vals = np.unique(x[r, :])
            if len(vals) == 1 and vals[0] != 0:
                top = x[:r, :]
                bottom = x[r + 1:, :]
                if top.shape == bottom.shape and top.size > 0:
                    return ("horizontal", r, vals[0])

        # Check cols
        for c in range(w):
            vals = np.unique(x[:, c])
            if len(vals) == 1 and vals[0] != 0:
                left = x[:, :c]
                right = x[:, c + 1:]
                if left.shape == right.shape and left.size > 0:
                    return ("vertical", c, vals[0])

        return None

    def _split_by_separator(self, x):
        sep = self._find_separator(x)
        if sep is None:
            return None

        orientation, idx, _ = sep

        if orientation == "horizontal":
            a = x[:idx, :]
            b = x[idx + 1:, :]
        else:
            a = x[:, :idx]
            b = x[:, idx + 1:]

        if a.shape != b.shape:
            return None

        return a, b

    def _make_panel_rule(self, train_pairs, mode="and"):
        output_color = None

        for inp, out in train_pairs:
            split = self._split_by_separator(inp)
            if split is None:
                return None

            a, b = split

            if a.shape != out.shape:
                return None

            if mode == "and":
                mask = (a != 0) & (b != 0)
            elif mode == "nor":
                mask = (a == 0) & (b == 0)
            else:
                return None

            nonzero_out = np.unique(out[out != 0])

            # Output should usually use exactly one nonzero color
            if len(nonzero_out) > 1:
                return None

            inferred = 1 if len(nonzero_out) == 0 else int(nonzero_out[0])

            candidate = np.zeros_like(a)
            candidate[mask] = inferred

            if not np.array_equal(candidate, out):
                return None

            if output_color is None:
                output_color = inferred
            elif output_color != inferred:
                return None

        if output_color is None:
            return None

        def rule(x):
            split = self._split_by_separator(x)
            if split is None:
                return None

            a, b = split

            if mode == "and":
                mask = (a != 0) & (b != 0)
            elif mode == "nor":
                mask = (a == 0) & (b == 0)
            else:
                return None

            out = np.zeros_like(a)
            out[mask] = output_color
            return out

        return rule