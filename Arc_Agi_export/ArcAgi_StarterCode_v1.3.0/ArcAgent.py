import numpy as np
from ArcProblem import ArcProblem


class ArcAgent:
    def __init__(self):
        self.debug = False
        self.debug_problem = None

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        problem_name = arc_problem.problem_name()

        train_pairs = []
        for s in arc_problem.training_set():
            inp = s.get_input_data().data()
            out = s.get_output_data().data()
            train_pairs.append((inp, out))

        test_input = arc_problem.test_set().get_input_data().data()
        predictions = []

        if self._should_debug(problem_name):
            print(f"\n===== Problem: {problem_name} =====")

        simple_rules = [
            ("identity", lambda x: x.copy()),
            ("rot90", lambda x: np.rot90(x, 1)),
            ("rot180", lambda x: np.rot90(x, 2)),
            ("rot270", lambda x: np.rot90(x, 3)),
            ("flipud", lambda x: np.flipud(x)),
            ("fliplr", lambda x: np.fliplr(x)),
            ("transpose", lambda x: x.T),
            ("crop_nonzero", self._crop_nonzero),
            ("crop_and_swap_two_colors", self._crop_and_swap_two_colors),
            ("quad_mirror", self._quad_mirror),
            ("pair_to_line_same_row", self._pair_to_line_same_row),
            ("pair_to_line_same_col", self._pair_to_line_same_col),
            ("inner_shape_recolor_from_four_markers", self._inner_shape_recolor_from_four_markers),
            ("extend_diag_from_1_and_2_blocks", self._extend_diag_from_1_and_2_blocks),
            ("sort_colors_by_frequency_vertical", self._sort_colors_by_frequency_vertical),
            ("fill_closed_barrier_with_majority_color", self._fill_closed_barrier_with_majority_color),
        ]

        for name, rule in simple_rules:
            try:
                fits = self._fits_all_training(train_pairs, rule, rule_name=name, problem_name=problem_name)
                if self._should_debug(problem_name):
                    print(f"[simple] {name}: fits={fits}")
                if fits:
                    pred = rule(test_input)
                    if self._should_debug(problem_name):
                        print(f"[simple] {name}: pred is None? {pred is None}")
                        if pred is not None:
                            print(f"[simple] {name}: pred shape={pred.shape}")
                            print(pred.tolist())
                    self._append_if_new(predictions, pred)
            except Exception as e:
                if self._should_debug(problem_name):
                    print(f"[simple] {name} crashed: {e}")

        color_rule = self._make_extract_single_input_color_rule(train_pairs, problem_name)
        if self._should_debug(problem_name):
            print("[learned] extract_single_input_color_rule is None?", color_rule is None)
        if color_rule is not None:
            try:
                pred = color_rule(test_input)
                if self._should_debug(problem_name):
                    print("[learned] extract_single_input_color_rule pred is None?", pred is None)
                    if pred is not None:
                        print(pred.tolist())
                self._append_if_new(predictions, pred)
            except Exception as e:
                if self._should_debug(problem_name):
                    print("[learned] extract_single_input_color_rule crashed:", e)

        binary_or_rule = self._make_binary_or_panel_rule(train_pairs, problem_name)
        if self._should_debug(problem_name):
            print("[learned] binary_or_panel_rule is None?", binary_or_rule is None)
        if binary_or_rule is not None:
            try:
                pred = binary_or_rule(test_input)
                if self._should_debug(problem_name):
                    print("[learned] binary_or_panel_rule pred is None?", pred is None)
                    if pred is not None:
                        print(pred.tolist())
                self._append_if_new(predictions, pred)
            except Exception as e:
                if self._should_debug(problem_name):
                    print("[learned] binary_or_panel_rule crashed:", e)

        for mode in ["and", "or", "xor", "nor"]:
            rule = self._make_panel_rule(train_pairs, mode, problem_name)
            if self._should_debug(problem_name):
                print(f"[panel] {mode}: rule is None? {rule is None}")
            if rule is not None:
                try:
                    pred = rule(test_input)
                    if self._should_debug(problem_name):
                        print(f"[panel] {mode}: pred is None? {pred is None}")
                        if pred is not None:
                            print(pred.tolist())
                    self._append_if_new(predictions, pred)
                except Exception as e:
                    if self._should_debug(problem_name):
                        print(f"[panel] {mode} crashed: {e}")

        if self._should_debug(problem_name):
            print(f"total predictions: {len(predictions)}")

        return predictions[:3]

    # -------------------------
    # debug helpers
    # -------------------------
    def _should_debug(self, problem_name):
        if not self.debug:
            return False
        if self.debug_problem is None:
            return True
        return problem_name == self.debug_problem

    def _debug_train_failure(self, problem_name, rule_name, idx, inp, out, pred):
        if not self._should_debug(problem_name):
            return

        print(f"  -> FAIL on train example {idx} for rule [{rule_name}]")
        if pred is None:
            print("     pred is None")
            return

        print(f"     pred shape = {pred.shape}, out shape = {out.shape}")
        print("     pred:")
        print(pred.tolist())
        print("     out :")
        print(out.tolist())

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

    def _fits_all_training(self, train_pairs, fn, rule_name="", problem_name=""):
        for i, (inp, out) in enumerate(train_pairs):
            pred = fn(inp)
            if pred is None or not np.array_equal(pred, out):
                self._debug_train_failure(problem_name, rule_name, i, inp, out, pred)
                return False
        return True

    def _tight_crop(self, x):
        coords = np.argwhere(x != 0)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return x[r0:r1 + 1, c0:c1 + 1]

    # -------------------------
    # basic rules
    # -------------------------
    def _crop_nonzero(self, x):
        return self._tight_crop(x)

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
        top = np.hstack([x, np.fliplr(x)])
        bottom = np.hstack([np.flipud(x), np.rot90(x, 2)])
        return np.vstack([top, bottom])

    def _neighbors4(self, r, c, h, w):
        if r > 0:
            yield r - 1, c
        if r + 1 < h:
            yield r + 1, c
        if c > 0:
            yield r, c - 1
        if c + 1 < w:
            yield r, c + 1

    def _connected_components_of_color(self, x, color):
        h, w = x.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        for r in range(h):
            for c in range(w):
                if visited[r, c] or x[r, c] != color:
                    continue

                stack = [(r, c)]
                visited[r, c] = True
                comp = []

                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for nr, nc in self._neighbors4(cr, cc, h, w):
                        if not visited[nr, nc] and x[nr, nc] == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                components.append(comp)

        return components

    def _fill_closed_barrier_with_majority_color(self, x):
        out = x.copy()
        changed = False
        h, w = x.shape

        for barrier_color in [int(c) for c in np.unique(x) if c != 0]:
            for component in self._connected_components_of_color(x, barrier_color):
                rows = [r for r, _ in component]
                cols = [c for _, c in component]
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)

                if r1 - r0 < 2 or c1 - c0 < 2:
                    continue

                comp_set = set(component)
                box_h = r1 - r0 + 1
                box_w = c1 - c0 + 1
                visited = np.zeros((box_h, box_w), dtype=bool)

                for br in range(box_h):
                    for bc in range(box_w):
                        gr, gc = r0 + br, c0 + bc
                        if (gr, gc) in comp_set or visited[br, bc]:
                            continue

                        stack = [(br, bc)]
                        visited[br, bc] = True
                        region = []
                        touches_box_edge = False

                        while stack:
                            cr, cc = stack.pop()
                            gr, gc = r0 + cr, c0 + cc
                            region.append((gr, gc))

                            if cr == 0 or cr == box_h - 1 or cc == 0 or cc == box_w - 1:
                                touches_box_edge = True

                            for nr, nc in self._neighbors4(cr, cc, box_h, box_w):
                                ngr, ngc = r0 + nr, c0 + nc
                                if visited[nr, nc] or (ngr, ngc) in comp_set:
                                    continue
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                        region_colors = [int(x[gr, gc]) for gr, gc in region
                                         if x[gr, gc] != 0 and x[gr, gc] != barrier_color]

                        if touches_box_edge:
                            for gr, gc in region:
                                if out[gr, gc] != barrier_color:
                                    out[gr, gc] = 0
                                    if x[gr, gc] != 0:
                                        changed = True
                            continue

                        if not region_colors:
                            continue

                        counts = {}
                        for color in region_colors:
                            counts[color] = counts.get(color, 0) + 1
                        fill_color = min(
                            [color for color, cnt in counts.items() if cnt == max(counts.values())]
                        )

                        for gr, gc in region:
                            if out[gr, gc] != barrier_color:
                                out[gr, gc] = fill_color
                                if x[gr, gc] != fill_color:
                                    changed = True

        return out if changed else None

    # -------------------------
    # pair -> line rules
    # -------------------------
    def _pair_to_line_same_row(self, x):
        out = np.zeros_like(x)
        colors = [int(c) for c in np.unique(x) if c != 0]

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
                    return None
                else:
                    return None
            else:
                return None

        return out if changed else None

    def _pair_to_line_same_col(self, x):
        out = np.zeros_like(x)
        colors = [int(c) for c in np.unique(x) if c != 0]

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
    # four-marker inner recolor rule
    # -------------------------
    def _inner_shape_recolor_from_four_markers(self, x):
        nonzero_colors = [int(c) for c in np.unique(x) if c != 0]

        for marker_color in nonzero_colors:
            coords = np.argwhere(x == marker_color)
            if len(coords) != 4:
                continue

            rows = sorted(set(int(r) for r, _ in coords))
            cols = sorted(set(int(c) for _, c in coords))

            if len(rows) != 2 or len(cols) != 2:
                continue

            r0, r1 = rows
            c0, c1 = cols

            expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            actual = set((int(r), int(c)) for r, c in coords)
            if actual != expected:
                continue

            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue

            interior = x[r0 + 1:r1, c0 + 1:c1].copy()

            mask = (interior != 0) & (interior != marker_color)
            if not np.any(mask):
                continue

            interior[mask] = marker_color
            return interior

        return None

    # -------------------------
    # diagonal extension rule for 1 and 2
    # -------------------------
    def _extend_diag_from_1_and_2_blocks(self, x):
        out = x.copy()

        # color 1: top-left corner -> (-1, -1)
        coords1 = np.argwhere(x == 1)
        if len(coords1) == 4:
            rows1 = sorted(set(int(r) for r, _ in coords1))
            cols1 = sorted(set(int(c) for _, c in coords1))
            if len(rows1) == 2 and len(cols1) == 2:
                r0, r1 = rows1
                c0, c1 = cols1
                expected1 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                actual1 = set((int(r), int(c)) for r, c in coords1)

                if actual1 == expected1:
                    r, c = r0 - 1, c0 - 1
                    while 0 <= r < x.shape[0] and 0 <= c < x.shape[1]:
                        out[r, c] = 1
                        r -= 1
                        c -= 1

        # color 2: bottom-right corner -> (+1, +1)
        coords2 = np.argwhere(x == 2)
        if len(coords2) == 4:
            rows2 = sorted(set(int(r) for r, _ in coords2))
            cols2 = sorted(set(int(c) for _, c in coords2))
            if len(rows2) == 2 and len(cols2) == 2:
                r0, r1 = rows2
                c0, c1 = cols2
                expected2 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                actual2 = set((int(r), int(c)) for r, c in coords2)

                if actual2 == expected2:
                    r, c = r1 + 1, c1 + 1
                    while 0 <= r < x.shape[0] and 0 <= c < x.shape[1]:
                        out[r, c] = 2
                        r += 1
                        c += 1

        return out

    # -------------------------
    # frequency sort rule
    # -------------------------
    def _sort_colors_by_frequency_vertical(self, x):

        colors = [int(c) for c in np.unique(x) if c != 0]
        if not colors:
            return None

        counts = []
        for color in colors:
            cnt = int(np.sum(x == color))
            counts.append((color, cnt))

        # count desc, same count면 color asc
        counts.sort(key=lambda t: (-t[1], t[0]))

        max_count = counts[0][1]
        num_colors = len(counts)

        out = np.zeros((max_count, num_colors), dtype=x.dtype)

        for col_idx, (color, cnt) in enumerate(counts):
            out[:cnt, col_idx] = color

        return out

    # -------------------------
    # learned color extraction
    # -------------------------
    def _make_extract_single_input_color_rule(self, train_pairs, problem_name=""):
        chosen_input_color = None
        chosen_output_color = None

        for idx, (inp, out) in enumerate(train_pairs):
            out_colors = [int(c) for c in np.unique(out) if c != 0]

            if len(out_colors) != 1:
                if self._should_debug(problem_name):
                    print(f"  -> extract_single_input_color_rule fail at train {idx}: output has {len(out_colors)} nonzero colors")
                return None

            out_color = out_colors[0]
            candidate_input_colors = [int(c) for c in np.unique(inp) if c != 0]

            matched_input_color = None
            for in_color in candidate_input_colors:
                candidate = np.zeros_like(inp)
                candidate[inp == in_color] = out_color
                if np.array_equal(candidate, out):
                    matched_input_color = in_color
                    break

            if matched_input_color is None:
                if self._should_debug(problem_name):
                    print(f"  -> extract_single_input_color_rule fail at train {idx}: no matching input color")
                return None

            if chosen_input_color is None:
                chosen_input_color = matched_input_color
                chosen_output_color = out_color
            elif chosen_input_color != matched_input_color or chosen_output_color != out_color:
                if self._should_debug(problem_name):
                    print(f"  -> extract_single_input_color_rule fail at train {idx}: inconsistent chosen colors")
                return None

        if chosen_input_color is None:
            return None

        def rule(x):
            if chosen_input_color not in np.unique(x):
                return None
            out = np.zeros_like(x)
            out[x == chosen_input_color] = chosen_output_color
            return out

        return rule

    def _make_binary_or_panel_rule(self, train_pairs, problem_name=""):
        for idx, (inp, out) in enumerate(train_pairs):
            split = self._split_by_separator(inp)
            if split is None:
                if self._should_debug(problem_name):
                    print(f"  -> binary_or_panel fail at train {idx}: no separator")
                return None

            left, right = split
            if left.shape != right.shape or left.shape != out.shape:
                if self._should_debug(problem_name):
                    print(f"  -> binary_or_panel fail at train {idx}: shape mismatch")
                return None

            candidate = ((left != 0) | (right != 0)).astype(inp.dtype)
            if not np.array_equal(candidate, out):
                if self._should_debug(problem_name):
                    print(f"  -> binary_or_panel fail at train {idx}: candidate mismatch")
                return None

        def rule(x):
            split = self._split_by_separator(x)
            if split is None:
                return None

            left, right = split
            if left.shape != right.shape:
                return None

            return ((left != 0) | (right != 0)).astype(x.dtype)

        return rule

    # -------------------------
    # separator panel rules
    # -------------------------
    def _find_separator(self, x):
        h, w = x.shape

        for r in range(h):
            vals = np.unique(x[r, :])
            if len(vals) == 1 and vals[0] != 0:
                a = x[:r, :]
                b = x[r + 1:, :]
                if a.shape == b.shape and a.size > 0:
                    return ("horizontal", r)

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
            return (a != 0) ^ (b != 0)
        if mode == "nor":
            return (a == 0) & (b == 0)
        return None

    def _make_panel_rule(self, train_pairs, mode, problem_name=""):
        out_color = None

        for idx, (inp, out) in enumerate(train_pairs):
            split = self._split_by_separator(inp)
            if split is None:
                if self._should_debug(problem_name):
                    print(f"  -> panel {mode} fail at train {idx}: no separator")
                return None

            a, b = split
            if a.shape != out.shape:
                if self._should_debug(problem_name):
                    print(f"  -> panel {mode} fail at train {idx}: shape mismatch {a.shape} vs {out.shape}")
                return None

            mask = self._bool_mask(a, b, mode)
            if mask is None:
                return None

            nz = np.unique(out[out != 0])
            if len(nz) > 1:
                if self._should_debug(problem_name):
                    print(f"  -> panel {mode} fail at train {idx}: output has multiple nonzero colors")
                return None

            inferred_color = 1 if len(nz) == 0 else int(nz[0])

            candidate = np.zeros_like(a)
            candidate[mask] = inferred_color

            if not np.array_equal(candidate, out):
                if self._should_debug(problem_name):
                    print(f"  -> panel {mode} fail at train {idx}: candidate mismatch")
                return None

            if out_color is None:
                out_color = inferred_color
            elif out_color != inferred_color:
                if self._should_debug(problem_name):
                    print(f"  -> panel {mode} fail at train {idx}: inconsistent output color")
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
