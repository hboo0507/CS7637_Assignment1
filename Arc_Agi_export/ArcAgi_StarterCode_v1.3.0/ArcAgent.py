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

        # ---- Try crop rule ----
        if self._fits_all_training(train_pairs, self._crop_nonzero):
            return [self._crop_nonzero(test_input)]

        # ---- Try AND rule ----
        if self._fits_all_training(train_pairs, self._two_panel_and):
            return [self._two_panel_and(test_input)]

        # ---- Fallback ----
        return [
            np.rot90(test_input, 1),
            np.rot90(test_input, 2),
            np.flipud(test_input)
        ][:3]

    # ----------------------------------------
    # Check rule on all training pairs
    # ----------------------------------------
    def _fits_all_training(self, train_pairs, fn):
        for inp, out in train_pairs:
            pred = fn(inp)
            if pred is None or not np.array_equal(pred, out):
                return False
        return True

    # ----------------------------------------
    # Tight crop around non-zero
    # ----------------------------------------
    def _crop_nonzero(self, x):
        coords = np.argwhere(x != 0)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return x[r0:r1+1, c0:c1+1]

    # ----------------------------------------
    # Two panel AND (separator = 5)
    # ----------------------------------------
    def _two_panel_and(self, x):
        sep_cols = [c for c in range(x.shape[1]) if np.all(x[:, c] == 5)]
        if not sep_cols:
            return None

        sep = sep_cols[0]

        left = x[:, :sep]
        right = x[:, sep+1:]

        if left.shape != right.shape:
            return None

        mask = (left != 0) & (right != 0)

        out = np.zeros_like(left)
        out[mask] = 2
        return out