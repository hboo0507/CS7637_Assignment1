import numpy as np
from ArcProblem import ArcProblem


class ArcAgent:
    def __init__(self):
        # Rule-based ARC agent
        # Uses training data to decide which transformation rule applies
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:

        # Collect training input-output pairs
        train_pairs = []
        for s in arc_problem.training_set():
            inp = s.get_input_data().data()
            out = s.get_output_data().data()
            train_pairs.append((inp, out))

        test_input = arc_problem.test_set().get_input_data().data()

        predictions = []

        # Candidate transformation rules
        rules = [
            self._crop_nonzero,
            self._two_panel_and,
            self._fill_diagonals,
            self._two_panel_nor_horizontal,
            lambda x: np.rot90(x, 1),
            lambda x: np.rot90(x, 2),
            lambda x: np.flipud(x),
        ]

        # Only keep rules that match ALL training examples
        for rule in rules:
            if self._fits_all_training(train_pairs, rule):
                pred = rule(test_input)
                if pred is not None:
                    predictions.append(pred)

            if len(predictions) == 3:
                break

        return predictions[:3]

    def _fits_all_training(self, train_pairs, fn):
        # Check if a rule reproduces every training output
        for inp, out in train_pairs:
            pred = fn(inp)
            if pred is None or not np.array_equal(pred, out):
                return False
        return True

    def _crop_nonzero(self, x):
        # Extract the smallest rectangle containing all non-zero pixels
        coords = np.argwhere(x != 0)
        if coords.size == 0:
            return None

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)

        return x[r0:r1+1, c0:c1+1]

    def _two_panel_and(self, x):
        # Split left/right using column of 5s
        # Output 2 where both sides are non-zero
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

    def _fill_diagonals(self, x):
        # Extend each non-zero pixel in four diagonal directions:
        # down-right, up-left, up-right, down-left
        h, w = x.shape
        out = x.copy()

        for r in range(h):
            for c in range(w):
                if x[r, c] != 0:
                    color = x[r, c]

                    # down-right
                    rr, cc = r, c
                    while rr < h and cc < w:
                        out[rr, cc] = color
                        rr += 1
                        cc += 1

                    # up-left
                    rr, cc = r, c
                    while rr >= 0 and cc >= 0:
                        out[rr, cc] = color
                        rr -= 1
                        cc -= 1

                    # up-right
                    rr, cc = r, c
                    while rr >= 0 and cc < w:
                        out[rr, cc] = color
                        rr -= 1
                        cc += 1

                    # down-left
                    rr, cc = r, c
                    while rr < h and cc >= 0:
                        out[rr, cc] = color
                        rr += 1
                        cc -= 1

        return out

    def _two_panel_nor_horizontal(self, x):
        # Split top/bottom using row of 4s
        # Output 3 where BOTH cells are zero (NOR logic)
        sep_rows = [r for r in range(x.shape[0]) if np.all(x[r, :] == 4)]
        if not sep_rows:
            return None

        sep = sep_rows[0]

        top = x[:sep, :]
        bottom = x[sep+1:, :]

        if top.shape != bottom.shape:
            return None

        mask = (top == 0) & (bottom == 0)

        out = np.zeros_like(top)
        out[mask] = 3

        return out