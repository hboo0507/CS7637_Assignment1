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
            ("fold_across_five_separator_row", self._fold_across_five_separator_row),
            ("decorate_four_rectangle_corners_and_connect", self._decorate_four_rectangle_corners_and_connect),
            ("expand_leading_run_to_half_width_staircase", self._expand_leading_run_to_half_width_staircase),
            ("fill_zero_regions_outer_three_inner_two", self._fill_zero_regions_outer_three_inner_two),
            ("fill_rows_when_endpoints_match", self._fill_rows_when_endpoints_match),
            ("recolor_holey_one_components_to_eight", self._recolor_holey_one_components_to_eight),
            ("overlay_three_equal_panels_left_to_right_priority", self._overlay_three_equal_panels_left_to_right_priority),
            ("extend_single_marker_from_wedge_apex", self._extend_single_marker_from_wedge_apex),
            ("move_three_one_step_toward_four", self._move_three_one_step_toward_four),
            ("trace_threes_between_single_one_and_two", self._trace_threes_between_single_one_and_two),
            ("surround_holey_ones_with_twos_and_fill_inner_edge_with_threes", self._surround_holey_ones_with_twos_and_fill_inner_edge_with_threes),
            ("count_panel_blocks_to_staircase", self._count_panel_blocks_to_staircase),
            ("marker_frame_to_small_staircase", self._marker_frame_to_small_staircase),
            ("recolor_cropped_shape_from_clue_pairs", self._recolor_cropped_shape_from_clue_pairs),
            ("expand_single_two_to_v_with_inner_diagonals", self._expand_single_two_to_v_with_inner_diagonals),
            ("draw_line_between_twos_with_one_to_three", self._draw_line_between_twos_with_one_to_three),
            ("complete_missing_panel_by_separator_mirroring", self._complete_missing_panel_by_separator_mirroring),
            ("mirror_propagate_across_full_separators", self._mirror_propagate_across_full_separators),
            ("xor_top_bottom_masks_to_six", self._xor_top_bottom_masks_to_six),
            ("overlay_split_panels_left_priority", self._overlay_split_panels_left_priority),
            ("fit_pieces_into_base_slots", self._fit_pieces_into_base_slots),
            ("mosaic_rot180_fliplr_bands", self._mosaic_rot180_fliplr_bands),
            ("mirror_attach_inside_8_border", self._mirror_attach_inside_8_border),
            ("fill_closed_barrier_with_majority_color", self._fill_closed_barrier_with_majority_color),
            ("connect_flower_centers_with_ones", self._connect_flower_centers_with_ones),
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

        marker_shape_rule = self._make_marker_shape_by_nearest_fold_signature_rule(train_pairs, problem_name)
        if self._should_debug(problem_name):
            print("[learned] marker_shape_by_nearest_fold_signature_rule is None?", marker_shape_rule is None)
        if marker_shape_rule is not None:
            try:
                pred = marker_shape_rule(test_input)
                if self._should_debug(problem_name):
                    print("[learned] marker_shape_by_nearest_fold_signature_rule pred is None?", pred is None)
                    if pred is not None:
                        print(pred.tolist())
                self._append_if_new(predictions, pred)
            except Exception as e:
                if self._should_debug(problem_name):
                    print("[learned] marker_shape_by_nearest_fold_signature_rule crashed:", e)

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

    def _mosaic_rot180_fliplr_bands(self, x):
        if x.shape != (3, 3):
            return None

        rot180 = np.rot90(x, 2)
        flipud = np.flipud(x)
        fliplr = np.fliplr(x)

        top = np.hstack([rot180, flipud, rot180])
        middle = np.hstack([fliplr, x, fliplr])
        bottom = np.hstack([rot180, flipud, rot180])
        return np.vstack([top, middle, bottom])

    def _xor_top_bottom_masks_to_six(self, x):
        h, w = x.shape
        if h != 6:
            return None

        top = x[:3, :]
        bottom = x[3:, :]
        top_mask = top != 0
        bottom_mask = bottom != 0
        xor_mask = np.logical_xor(top_mask, bottom_mask)

        out = np.zeros((3, w), dtype=x.dtype)
        out[xor_mask] = 6
        return out

    def _overlay_split_panels_left_priority(self, x):
        h, w = x.shape
        if w < 3:
            return None

        sep_candidates = []
        for c in range(w):
            col = x[:, c]
            if np.all(col == col[0]) and col[0] != 0:
                sep_candidates.append(c)
        if not sep_candidates:
            return None

        sep_options = [c for c in sep_candidates if c > 0 and c + 1 < w and c == w - c - 1]
        if len(sep_options) != 1:
            return None
        sep = sep_options[0]
        left = x[:, :sep]
        right = x[:, sep + 1:]
        if left.shape != right.shape:
            return None

        out = left.copy()
        right_mask = right != 0
        left_zero_mask = left == 0
        if np.array_equal(right_mask, left_zero_mask):
            out[right_mask] = right[right_mask]
        return out

    def _mirror_propagate_across_full_separators(self, x):
        h, w = x.shape
        colors = [int(c) for c in np.unique(x) if c != 0]
        if not colors:
            return None

        sep_color = max(colors, key=lambda c: int(np.sum(x == c)))
        sep_rows = [r for r in range(h) if np.all(x[r, :] == sep_color)]
        sep_cols = [c for c in range(w) if np.all(x[:, c] == sep_color)]
        if not sep_rows and not sep_cols:
            return None

        out = x.copy()
        changed = False

        row_edges = [-1] + sep_rows + [h]
        row_spans = [(row_edges[i] + 1, row_edges[i + 1]) for i in range(len(row_edges) - 1)]

        col_edges = [-1] + sep_cols + [w]
        col_spans = [(col_edges[i] + 1, col_edges[i + 1]) for i in range(len(col_edges) - 1)]

        for sep_idx, sc in enumerate(sep_cols, start=1):
            left_c0, left_c1 = col_spans[sep_idx - 1]
            right_c0, right_c1 = col_spans[sep_idx]
            left_w = left_c1 - left_c0
            right_w = right_c1 - right_c0
            if left_w != right_w or left_w <= 0:
                continue

            reference_pairs = []
            for r0, r1 in row_spans:
                if r1 <= r0:
                    continue
                left_block = x[r0:r1, left_c0:left_c1]
                right_block = x[r0:r1, right_c0:right_c1]
                if np.any(left_block != 0) and np.any(right_block != 0) and np.array_equal(np.fliplr(left_block), right_block):
                    reference_pairs.append(left_block.copy())

            if not reference_pairs:
                continue

            for r0, r1 in row_spans:
                if r1 <= r0:
                    continue
                left_block = x[r0:r1, left_c0:left_c1]
                right_block = x[r0:r1, right_c0:right_c1]

                left_nonzero = np.any(left_block != 0)
                right_nonzero = np.any(right_block != 0)

                if left_nonzero and not right_nonzero:
                    if any(np.array_equal(left_block, ref) for ref in reference_pairs):
                        out[r0:r1, right_c0:right_c1] = np.fliplr(left_block)
                        changed = True
                elif right_nonzero and not left_nonzero:
                    if any(np.array_equal(np.fliplr(right_block), ref) for ref in reference_pairs):
                        out[r0:r1, left_c0:left_c1] = np.fliplr(right_block)
                        changed = True

        return out if changed else None

    def _complete_missing_panel_by_separator_mirroring(self, x):
        h, w = x.shape
        colors = [int(c) for c in np.unique(x) if c != 0]
        if not colors:
            return None

        sep_color = max(colors, key=lambda c: int(np.sum(x == c)))
        sep_rows = [r for r in range(h) if np.all(x[r, :] == sep_color)]
        sep_cols = [c for c in range(w) if np.all(x[:, c] == sep_color)]
        if not sep_rows or not sep_cols:
            return None

        row_edges = [-1] + sep_rows + [h]
        col_edges = [-1] + sep_cols + [w]
        row_spans = [(row_edges[i] + 1, row_edges[i + 1]) for i in range(len(row_edges) - 1)]
        col_spans = [(col_edges[i] + 1, col_edges[i + 1]) for i in range(len(col_edges) - 1)]

        out = x.copy()
        changed = False

        def get_block(ri, ci):
            r0, r1 = row_spans[ri]
            c0, c1 = col_spans[ci]
            if r0 >= r1 or c0 >= c1:
                return None
            return x[r0:r1, c0:c1]

        def set_block(ri, ci, block):
            r0, r1 = row_spans[ri]
            c0, c1 = col_spans[ci]
            out[r0:r1, c0:c1] = block

        def nonzero(block):
            return block is not None and np.any(block != 0)

        for ri in range(len(row_spans) - 1):
            for ci in range(len(col_spans) - 1):
                tl = get_block(ri, ci)
                tr = get_block(ri, ci + 1)
                bl = get_block(ri + 1, ci)
                br = get_block(ri + 1, ci + 1)
                blocks = [tl, tr, bl, br]
                present = [nonzero(b) for b in blocks]
                if sum(present) != 3:
                    continue

                missing = present.index(False)
                candidates = []

                if missing == 0:
                    if nonzero(tr):
                        candidates.append(np.fliplr(tr))
                    if nonzero(bl):
                        candidates.append(np.flipud(bl))
                    if nonzero(br):
                        candidates.append(np.flipud(np.fliplr(br)))
                elif missing == 1:
                    if nonzero(tl):
                        candidates.append(np.fliplr(tl))
                    if nonzero(br):
                        candidates.append(np.flipud(br))
                    if nonzero(bl):
                        candidates.append(np.flipud(np.fliplr(bl)))
                elif missing == 2:
                    if nonzero(tl):
                        candidates.append(np.flipud(tl))
                    if nonzero(br):
                        candidates.append(np.fliplr(br))
                    if nonzero(tr):
                        candidates.append(np.flipud(np.fliplr(tr)))
                else:
                    if nonzero(tr):
                        candidates.append(np.flipud(tr))
                    if nonzero(bl):
                        candidates.append(np.fliplr(bl))
                    if nonzero(tl):
                        candidates.append(np.flipud(np.fliplr(tl)))

                if not candidates:
                    continue

                target = candidates[0]
                if any(c.shape != target.shape or not np.array_equal(c, target) for c in candidates[1:]):
                    continue

                if missing == 0:
                    set_block(ri, ci, target)
                elif missing == 1:
                    set_block(ri, ci + 1, target)
                elif missing == 2:
                    set_block(ri + 1, ci, target)
                else:
                    set_block(ri + 1, ci + 1, target)
                changed = True

        return out if changed else None

    def _unique_orientations(self, piece):
        variants = []
        seen = set()

        candidates = [
            piece,
            np.rot90(piece, 1),
            np.rot90(piece, 2),
            np.rot90(piece, 3),
            np.fliplr(piece),
            np.flipud(piece),
            np.rot90(np.fliplr(piece), 1),
            np.rot90(np.flipud(piece), 1),
        ]

        for cand in candidates:
            key = tuple(tuple(int(v) for v in row) for row in cand.tolist())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cand)

        return variants

    def _fit_pieces_into_base_slots(self, x):
        h, w = x.shape
        all_components = self._all_nonzero_components(x)
        if len(all_components) < 2:
            return None

        base_color = None
        base_comp = None
        best_score = None
        for color, comp in all_components:
            rows = [r for r, _ in comp]
            if max(rows) != h - 1:
                continue
            score = (len(comp), color)
            if best_score is None or score > best_score:
                best_score = score
                base_color = color
                base_comp = comp

        if base_comp is None:
            return None

        base_set = set(base_comp)
        top_row = min(r for r, _ in base_comp)
        if top_row == 0:
            return None

        min_col = min(c for _, c in base_comp)
        max_col = max(c for _, c in base_comp)

        out = np.zeros_like(x)
        for r, c in base_comp:
            out[r, c] = base_color

        slots = []
        c = min_col
        while c <= max_col:
            if x[top_row, c] != 0:
                c += 1
                continue

            c0 = c
            while c <= max_col and x[top_row, c] == 0:
                c += 1
            c1 = c - 1

            if top_row - 1 < 0:
                return None
            if any(x[top_row - 1, cc] != 0 for cc in range(c0, c1 + 1)):
                return None

            slots.append((c0, c1))

        pieces = []
        for color, comp in all_components:
            if color == base_color and comp == base_comp:
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            piece = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=x.dtype)
            for rr, cc in comp:
                piece[rr - r0, cc - c0] = color
            pieces.append((color, comp, piece))

        changed = False
        pieces.sort(key=lambda item: -len(item[1]))
        slots = sorted(slots, key=lambda span: span[1] - span[0] + 1)
        used_pieces = set()

        for c0, c1 in slots:
            slot_w = c1 - c0 + 1
            matched = False

            for piece_idx, (color, comp, piece) in enumerate(pieces):
                if piece_idx in used_pieces:
                    continue

                for oriented in self._unique_orientations(piece):
                    occ = oriented != 0
                    oh, ow = occ.shape
                    if oh != 2 or ow != slot_w:
                        continue
                    if not np.all(occ):
                        continue

                    for dr in range(2):
                        for dc in range(slot_w):
                            out[top_row - 1 + dr, c0 + dc] = int(oriented[dr, dc])
                    used_pieces.add(piece_idx)
                    changed = True
                    matched = True
                    break

                if matched:
                    break

        return out if changed else None

    def _find_flower_centers(self, x, color=2):
        h, w = x.shape
        centers = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if x[r, c] != 0:
                    continue
                if x[r - 1, c] != color or x[r + 1, c] != color:
                    continue
                if x[r, c - 1] != color or x[r, c + 1] != color:
                    continue
                centers.append((r, c))
        return centers

    def _connect_flower_centers_with_ones(self, x):
        centers = self._find_flower_centers(x, color=2)
        if len(centers) < 2:
            return None

        out = x.copy()
        changed = False

        rows = {}
        cols = {}
        for r, c in centers:
            rows.setdefault(r, []).append(c)
            cols.setdefault(c, []).append(r)

        for r, cols_in_row in rows.items():
            cols_in_row.sort()
            for left, right in zip(cols_in_row, cols_in_row[1:]):
                for c in range(left + 1, right):
                    if out[r, c] == 0:
                        out[r, c] = 1
                        changed = True

        for c, rows_in_col in cols.items():
            rows_in_col.sort()
            for top, bottom in zip(rows_in_col, rows_in_col[1:]):
                for r in range(top + 1, bottom):
                    if out[r, c] == 0:
                        out[r, c] = 1
                        changed = True

        return out if changed else None

    def _mirror_attach_inside_8_border(self, x):
        out = x.copy()
        changed = False
        h, w = x.shape

        for comp in self._connected_components_of_color(x, 8):
            comp_set = set(comp)

            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            border_h = max(rows) - min(rows) + 1
            border_w = max(cols) - min(cols) + 1

            reachable = np.zeros((h, w), dtype=bool)
            stack = []

            for r in range(h):
                for c in [0, w - 1]:
                    if (r, c) not in comp_set and not reachable[r, c]:
                        reachable[r, c] = True
                        stack.append((r, c))
            for c in range(w):
                for r in [0, h - 1]:
                    if (r, c) not in comp_set and not reachable[r, c]:
                        reachable[r, c] = True
                        stack.append((r, c))

            while stack:
                r, c = stack.pop()
                for nr, nc in self._neighbors4(r, c, h, w):
                    if (nr, nc) in comp_set or reachable[nr, nc]:
                        continue
                    reachable[nr, nc] = True
                    stack.append((nr, nc))

            visited = np.zeros((h, w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    if (r, c) in comp_set or reachable[r, c] or visited[r, c]:
                        continue

                    region = []
                    region_stack = [(r, c)]
                    visited[r, c] = True

                    while region_stack:
                        cr, cc = region_stack.pop()
                        region.append((cr, cc))
                        for nr, nc in self._neighbors4(cr, cc, h, w):
                            if (nr, nc) in comp_set or reachable[nr, nc] or visited[nr, nc]:
                                continue
                            visited[nr, nc] = True
                            region_stack.append((nr, nc))

                    object_cells = [(rr, cc) for rr, cc in region if x[rr, cc] != 0 and x[rr, cc] != 8]
                    if not object_cells:
                        continue

                    reg_rows = [rr for rr, _ in region]
                    reg_cols = [cc for _, cc in region]
                    r0, r1 = min(reg_rows), max(reg_rows)
                    c0, c1 = min(reg_cols), max(reg_cols)

                    if border_w >= border_h:
                        center_sum = c0 + c1
                        for rr, cc in object_cells:
                            mirror_c = center_sum - cc
                            if (rr, mirror_c) in region and out[rr, mirror_c] == 0:
                                out[rr, mirror_c] = int(x[rr, cc])
                                changed = True
                    else:
                        center_sum = r0 + r1
                        for rr, cc in object_cells:
                            mirror_r = center_sum - rr
                            if (mirror_r, cc) in region and out[mirror_r, cc] == 0:
                                out[mirror_r, cc] = int(x[rr, cc])
                                changed = True

        return out if changed else None

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

    def _all_nonzero_components(self, x):
        components = []
        for color in [int(c) for c in np.unique(x) if c != 0]:
            for comp in self._connected_components_of_color(x, color):
                components.append((color, comp))
        return components

    def _fill_closed_barrier_with_majority_color(self, x):
        out = np.zeros_like(x)
        changed = False
        h, w = x.shape

        barrier_components = []
        for color, comp in self._all_nonzero_components(x):
            if len(comp) < 5:
                continue

            comp_set = set(comp)
            barrier_components.append((color, comp_set))
            for r, c in comp:
                out[r, c] = color
                if x[r, c] != color:
                    changed = True

        if not barrier_components:
            return None

        for barrier_color, comp_set in barrier_components:
            reachable = np.zeros((h, w), dtype=bool)
            stack = []

            for r in range(h):
                for c in [0, w - 1]:
                    if (r, c) not in comp_set and not reachable[r, c]:
                        reachable[r, c] = True
                        stack.append((r, c))
            for c in range(w):
                for r in [0, h - 1]:
                    if (r, c) not in comp_set and not reachable[r, c]:
                        reachable[r, c] = True
                        stack.append((r, c))

            while stack:
                r, c = stack.pop()
                for nr, nc in self._neighbors4(r, c, h, w):
                    if (nr, nc) in comp_set or reachable[nr, nc]:
                        continue
                    reachable[nr, nc] = True
                    stack.append((nr, nc))

            visited = np.zeros((h, w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    if (r, c) in comp_set or reachable[r, c] or visited[r, c]:
                        continue

                    region = []
                    region_stack = [(r, c)]
                    visited[r, c] = True

                    while region_stack:
                        cr, cc = region_stack.pop()
                        region.append((cr, cc))
                        for nr, nc in self._neighbors4(cr, cc, h, w):
                            if (nr, nc) in comp_set or reachable[nr, nc] or visited[nr, nc]:
                                continue
                            visited[nr, nc] = True
                            region_stack.append((nr, nc))

                    region_colors = [
                        int(x[rr, cc]) for rr, cc in region
                        if x[rr, cc] != 0 and (rr, cc) not in comp_set
                    ]
                    if not region_colors:
                        continue

                    counts = {}
                    for color in region_colors:
                        counts[color] = counts.get(color, 0) + 1
                    max_count = max(counts.values())
                    fill_color = min(color for color, cnt in counts.items() if cnt == max_count)

                    for rr, cc in region:
                        if out[rr, cc] != fill_color:
                            out[rr, cc] = fill_color
                            if x[rr, cc] != fill_color:
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
    # remove a full row of 5s and overlay the top and bottom halves
    # -------------------------
    def _fold_across_five_separator_row(self, x):
        h, w = x.shape
        five_rows = [r for r in range(h) if np.all(x[r, :] == 5)]
        if len(five_rows) != 1:
            return None

        sep = five_rows[0]
        top = x[:sep, :]
        bottom = x[sep + 1:, :]
        if top.shape != bottom.shape or top.shape[0] == 0:
            return None

        out = top.copy()
        mask = bottom != 0
        out[mask] = bottom[mask]
        return out

    # -------------------------
    # four corner markers define a rectangle:
    # expand each into a 3x3 block of the other color and connect the centers with 5s
    # -------------------------
    def _decorate_four_rectangle_corners_and_connect(self, x):
        pts = np.argwhere(x != 0)
        if len(pts) != 4:
            return None

        rows = sorted({int(r) for r, _ in pts})
        cols = sorted({int(c) for _, c in pts})
        if len(rows) != 2 or len(cols) != 2:
            return None

        r0, r1 = rows
        c0, c1 = cols
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if any(x[r, c] == 0 for r, c in corners):
            return None

        colors = sorted({int(x[r, c]) for r, c in corners})
        if len(colors) != 2:
            return None

        a, b = colors
        if int(x[r0, c0]) != int(x[r1, c1]) or int(x[r0, c1]) != int(x[r1, c0]):
            return None
        if {int(x[r0, c0]), int(x[r0, c1])} != {a, b}:
            return None

        h, w = x.shape
        if r0 - 1 < 0 or r1 + 1 >= h or c0 - 1 < 0 or c1 + 1 >= w:
            return None

        out = np.zeros_like(x)
        for r, c in corners:
            center_color = int(x[r, c])
            surround_color = b if center_color == a else a
            out[r - 1:r + 2, c - 1:c + 2] = surround_color
            out[r, c] = center_color

        for c in range(c0 + 2, c1 - 1):
            if min(c - (c0 + 2), (c1 - 2) - c) % 2 == 0:
                out[r0, c] = 5
                out[r1, c] = 5

        for r in range(r0 + 2, r1 - 1):
            if min(r - (r0 + 2), (r1 - 2) - r) % 2 == 0:
                out[r, c0] = 5
                out[r, c1] = 5
        return out

    # -------------------------
    # grow the leading nonzero run one step per row for half the width
    # -------------------------
    def _expand_leading_run_to_half_width_staircase(self, x):
        if x.shape[0] != 1 or x.shape[1] % 2 != 0:
            return None

        row = x[0]
        nz = np.flatnonzero(row != 0)
        if len(nz) == 0:
            return None

        run_len = len(nz)
        if not np.array_equal(nz, np.arange(run_len)):
            return None

        color = int(row[0])
        if np.any(row[:run_len] != color) or np.any(row[run_len:] != 0):
            return None

        h = x.shape[1] // 2
        out = np.zeros((h, x.shape[1]), dtype=x.dtype)
        for r in range(h):
            out[r, :run_len + r] = color
        return out

    # -------------------------
    # fill zero background with 3, but enclosed zero regions with 2
    # -------------------------
    def _fill_zero_regions_outer_three_inner_two(self, x):
        if np.any((x != 0) & (x == 2)):
            return None

        h, w = x.shape
        zero_seen = np.zeros((h, w), dtype=bool)
        out = x.copy()
        found_zero = False

        for r in range(h):
            for c in range(w):
                if x[r, c] != 0 or zero_seen[r, c]:
                    continue
                found_zero = True
                stack = [(r, c)]
                zero_seen[r, c] = True
                comp = []
                touches_border = False
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    if rr in (0, h - 1) or cc in (0, w - 1):
                        touches_border = True
                    for nr, nc in self._neighbors4(rr, cc, h, w):
                        if x[nr, nc] == 0 and not zero_seen[nr, nc]:
                            zero_seen[nr, nc] = True
                            stack.append((nr, nc))
                fill = 3 if touches_border else 2
                for rr, cc in comp:
                    out[rr, cc] = fill

        return out if found_zero else None

    # -------------------------
    # fill a whole row when its two endpoint markers match
    # -------------------------
    def _fill_rows_when_endpoints_match(self, x):
        h, w = x.shape
        if w < 2:
            return None
        out = x.copy()
        for r in range(h):
            left = int(x[r, 0])
            right = int(x[r, w - 1])
            if left != 0 and left == right and np.any(x[r, 1:w - 1] == 0):
                out[r, :] = left
        return out

    # -------------------------
    # recolor only 1-components that enclose at least one zero hole to 8
    # -------------------------
    def _recolor_holey_one_components_to_eight(self, x):
        comps = self._connected_components_of_color(x, 1)
        if not comps:
            return None

        h, w = x.shape
        out = x.copy()
        changed = False
        values, counts = np.unique(x, return_counts=True)
        bg_color = int(values[np.argmax(counts)])

        for comp in comps:
            comp_set = set(comp)
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)

            reachable = np.zeros((h, w), dtype=bool)
            stack = []
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if r in (r0, r1) or c in (c0, c1):
                        if x[r, c] == bg_color and (r, c) not in comp_set and not reachable[r, c]:
                            reachable[r, c] = True
                            stack.append((r, c))

            while stack:
                r, c = stack.pop()
                for nr, nc in self._neighbors4(r, c, h, w):
                    if nr < r0 or nr > r1 or nc < c0 or nc > c1:
                        continue
                    if x[nr, nc] != bg_color or (nr, nc) in comp_set or reachable[nr, nc]:
                        continue
                    reachable[nr, nc] = True
                    stack.append((nr, nc))

            has_hole = False
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if x[r, c] == bg_color and (r, c) not in comp_set and not reachable[r, c]:
                        has_hole = True
                        break
                if has_hole:
                    break

            if has_hole:
                for r, c in comp:
                    out[r, c] = 8
                    changed = True

        return out if changed else None

    # -------------------------
    # overlay three equal-width panels separated by full columns
    # using left-to-right nonzero priority
    # -------------------------
    def _overlay_three_equal_panels_left_to_right_priority(self, x):
        h, w = x.shape
        if h != 4 or w != 14:
            return None

        if not np.all(x[:, 4] == 2) or not np.all(x[:, 9] == 2):
            return None

        c1, c2 = 4, 9
        left = x[:, :c1]
        middle = x[:, c1 + 1:c2]
        right = x[:, c2 + 1:]
        if left.shape != middle.shape or middle.shape != right.shape:
            return None

        out = left.copy()
        mask = out == 0
        out[mask & (middle != 0)] = middle[mask & (middle != 0)]
        mask = out == 0
        out[mask & (right != 0)] = right[mask & (right != 0)]
        return out

    # -------------------------
    # extend the singleton marker color away from the apex side
    # of the surrounding wedge component
    # -------------------------
    def _extend_single_marker_from_wedge_apex(self, x):
        colors = [int(c) for c in np.unique(x) if c != 0]
        if len(colors) != 2:
            return None

        counts = {c: int(np.sum(x == c)) for c in colors}
        marker_color = next((c for c in colors if counts[c] == 1), None)
        if marker_color is None:
            return None
        bulk_color = next(c for c in colors if c != marker_color)

        r, c = map(int, np.argwhere(x == marker_color)[0])
        bulk = np.argwhere(x == bulk_color)
        if bulk.size == 0:
            return None

        r0, c0 = bulk.min(axis=0)
        r1, c1 = bulk.max(axis=0)
        if r == r0:
            dr, dc = 1, 0
        elif r == r1:
            dr, dc = -1, 0
        elif c == c0:
            dr, dc = 0, 1
        elif c == c1:
            dr, dc = 0, -1
        else:
            return None

        out = x.copy()
        cr, cc = r, c
        # Skip over the wedge itself, then extend only through the empty region.
        while 0 <= cr < x.shape[0] and 0 <= cc < x.shape[1] and out[cr, cc] != 0:
            cr += dr
            cc += dc
        while 0 <= cr < x.shape[0] and 0 <= cc < x.shape[1]:
            out[cr, cc] = marker_color
            cr += dr
            cc += dc
        return out

    # -------------------------
    # move the single 3 one king-step toward the single 4
    # -------------------------
    def _move_three_one_step_toward_four(self, x):
        threes = np.argwhere(x == 3)
        fours = np.argwhere(x == 4)
        others = [int(c) for c in np.unique(x) if c not in (0, 3, 4)]
        if len(threes) != 1 or len(fours) != 1 or others:
            return None

        r3, c3 = map(int, threes[0])
        r4, c4 = map(int, fours[0])
        dr = 0 if r4 == r3 else (1 if r4 > r3 else -1)
        dc = 0 if c4 == c3 else (1 if c4 > c3 else -1)
        nr, nc = r3 + dr, c3 + dc
        if not (0 <= nr < x.shape[0] and 0 <= nc < x.shape[1]):
            return None

        out = x.copy()
        out[r3, c3] = 0
        out[nr, nc] = 3
        return out

    # -------------------------
    # single 1 / single 2 path rule
    # -------------------------
    def _trace_threes_between_single_one_and_two(self, x):
        ones = np.argwhere(x == 1)
        twos = np.argwhere(x == 2)
        others = [int(c) for c in np.unique(x) if c not in (0, 1, 2)]
        if len(ones) != 1 or len(twos) != 1 or others:
            return None

        (r1, c1) = [int(v) for v in ones[0]]
        (r2, c2) = [int(v) for v in twos[0]]
        dr = 1 if r1 > r2 else -1 if r1 < r2 else 0
        dc = 1 if c1 > c2 else -1 if c1 < c2 else 0
        if dr == 0 and dc == 0:
            return None

        out = x.copy()
        cr, cc = r2, c2

        # Always take one diagonal step first when possible.
        if dr != 0 and dc != 0:
            cr += dr
            cc += dc
            if (cr, cc) != (r1, c1):
                out[cr, cc] = 3

        while True:
            rem_r = r1 - cr
            rem_c = c1 - cc
            if max(abs(rem_r), abs(rem_c)) <= 1:
                break

            if abs(rem_r) == abs(rem_c):
                cr += 1 if rem_r > 0 else -1
                cc += 1 if rem_c > 0 else -1
            elif abs(rem_r) > abs(rem_c):
                cr += 1 if rem_r > 0 else -1
            else:
                cc += 1 if rem_c > 0 else -1

            if (cr, cc) == (r1, c1):
                break
            out[cr, cc] = 3

        return out

    # -------------------------
    # add outer 2-ring to hole-bearing 1-components and fill
    # hole cells adjacent to the component with 3
    # -------------------------
    def _surround_holey_ones_with_twos_and_fill_inner_edge_with_threes(self, x):
        h, w = x.shape
        out = x.copy()
        changed = False

        comps = self._connected_components_of_color(x, 1)
        if not comps:
            return None

        for comp in comps:
            comp_set = set(comp)
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)

            reachable = np.zeros((h, w), dtype=bool)
            stack = []
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if r in (r0, r1) or c in (c0, c1):
                        if (r, c) not in comp_set and x[r, c] == 0 and not reachable[r, c]:
                            reachable[r, c] = True
                            stack.append((r, c))

            while stack:
                r, c = stack.pop()
                for nr, nc in self._neighbors4(r, c, h, w):
                    if nr < r0 or nr > r1 or nc < c0 or nc > c1:
                        continue
                    if (nr, nc) in comp_set or x[nr, nc] != 0 or reachable[nr, nc]:
                        continue
                    reachable[nr, nc] = True
                    stack.append((nr, nc))

            holes = []
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if x[r, c] == 0 and not reachable[r, c] and (r, c) not in comp_set:
                        holes.append((r, c))

            if not holes:
                continue

            # 8-neighbor outer ring.
            for r, c in comp:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in comp_set and x[nr, nc] == 0:
                            if out[nr, nc] == 0:
                                out[nr, nc] = 2
                                changed = True

            for r, c in holes:
                near_component = False
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        if (nr, nc) == (r, c):
                            continue
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) in comp_set:
                            near_component = True
                            break
                    if near_component:
                        break

                if near_component:
                    out[r, c] = 3
                    changed = True
                elif out[r, c] == 2:
                    out[r, c] = 0

        return out

    # -------------------------
    # count monochrome 2x2 blocks in separator panels -> staircase
    # -------------------------
    def _count_panel_blocks_to_staircase(self, x):
        colors = [int(c) for c in np.unique(x) if c != 0]
        if not colors:
            return None

        # The dominant separator color forms full rows/cols.
        sep_color = max(colors, key=lambda color: int(np.sum(x == color)))
        counts = {}

        h, w = x.shape
        for r in range(h - 1):
            for c in range(w - 1):
                block = x[r:r + 2, c:c + 2]
                vals = np.unique(block)
                if len(vals) != 1:
                    continue
                color = int(vals[0])
                if color == 0 or color == sep_color:
                    continue
                counts.setdefault(color, set()).add((r, c))

        if not counts:
            return None

        items = sorted(((color, len(pos)) for color, pos in counts.items()), key=lambda t: (t[1], t[0]))
        out = np.zeros((len(items), max(cnt for _, cnt in items)), dtype=x.dtype)
        for r, (color, cnt) in enumerate(items):
            out[r, :cnt] = color
        return out

    # -------------------------
    # single marker-color around a 1-frame -> small staircase summary
    # -------------------------
    def _marker_frame_to_small_staircase(self, x):
        marker_colors = [int(c) for c in np.unique(x) if c not in (0, 1)]
        if len(marker_colors) != 1:
            return None

        ones = np.argwhere(x == 1)
        if len(ones) == 0:
            return None

        r0, r1 = int(ones[:, 0].min()), int(ones[:, 0].max())
        c0, c1 = int(ones[:, 1].min()), int(ones[:, 1].max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        color = marker_colors[0]
        out = np.zeros((3, 3), dtype=x.dtype)
        out[0, :] = color

        if min(h, w) <= 4:
            return out
        if h == w:
            out[1, :2] = color
            return out

        out[1, 0] = color
        return out

    def _marker_signature(self, x):
        ones = np.argwhere(x == 1)
        if len(ones) == 0:
            return None

        marker_colors = [int(c) for c in np.unique(x) if c not in (0, 1)]
        if len(marker_colors) != 1:
            return None

        color = marker_colors[0]
        marks = np.argwhere(x == color)
        if len(marks) == 0:
            return None

        r0, r1 = ones[:, 0].min(), ones[:, 0].max()
        c0, c1 = ones[:, 1].min(), ones[:, 1].max()
        bins = {}
        for r, c in marks:
            vd = int(min(abs(r - r0), abs(r - r1)))
            hd = int(min(abs(c - c0), abs(c - c1)))
            bins[(vd, hd)] = bins.get((vd, hd), 0) + 1

        return color, tuple(sorted(bins.items()))

    def _make_marker_shape_by_nearest_fold_signature_rule(self, train_pairs, problem_name=""):
        prototypes = []

        for idx, (inp, out) in enumerate(train_pairs):
            sig = self._marker_signature(inp)
            if sig is None:
                return None
            color, bins = sig

            out_colors = [int(c) for c in np.unique(out) if c != 0]
            if len(out_colors) != 1 or out_colors[0] != color:
                return None

            coords = np.argwhere(out == color)
            if len(coords) == 0:
                return None

            pattern = np.zeros_like(out)
            pattern[out == color] = 1
            prototypes.append((bins, pattern))

        if not prototypes:
            return None

        def dist(a_bins, b_bins):
            a = dict(a_bins)
            b = dict(b_bins)
            keys = set(a) | set(b)
            return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)

        def rule(x):
            sig = self._marker_signature(x)
            if sig is None:
                return None
            color, bins = sig

            best_pattern = None
            best_score = None
            for proto_bins, proto_pattern in prototypes:
                score = dist(bins, proto_bins)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pattern = proto_pattern

            if best_pattern is None:
                return None

            out = np.zeros_like(best_pattern, dtype=x.dtype)
            out[best_pattern == 1] = color
            return out

        return rule

    # -------------------------
    # clue-pair recolor + crop rule
    # -------------------------
    def _largest_nonzero_component_bbox(self, x):
        h, w = x.shape
        visited = np.zeros((h, w), dtype=bool)
        best = None

        for r in range(h):
            for c in range(w):
                if visited[r, c] or x[r, c] == 0:
                    continue

                stack = [(r, c)]
                visited[r, c] = True
                comp = []

                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for nr, nc in self._neighbors4(cr, cc, h, w):
                        if not visited[nr, nc] and x[nr, nc] != 0:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                if best is None or len(comp) > len(best):
                    best = comp

        if not best:
            return None

        rows = [r for r, _ in best]
        cols = [c for _, c in best]
        return min(rows), max(rows), min(cols), max(cols)

    def _all_nonzero_connected_components(self, x):
        h, w = x.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        for r in range(h):
            for c in range(w):
                if visited[r, c] or x[r, c] == 0:
                    continue

                stack = [(r, c)]
                visited[r, c] = True
                comp = []

                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for nr, nc in self._neighbors4(cr, cc, h, w):
                        if not visited[nr, nc] and x[nr, nc] != 0:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                components.append(comp)

        return components

    def _recolor_cropped_shape_from_clue_pairs(self, x):
        bbox = self._largest_nonzero_component_bbox(x)
        if bbox is None:
            return None

        r0, r1, c0, c1 = bbox
        cropped = x[r0:r1 + 1, c0:c1 + 1].copy()

        mapping = {}
        for comp in self._all_nonzero_connected_components(x):
            if len(comp) != 2:
                continue

            if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in comp):
                continue

            cells = sorted(comp, key=lambda rc: (rc[0], rc[1]))
            (ra, ca), (rb, cb) = cells

            # Training cases use 2-cell horizontal clues: left color is target,
            # right color is source to be recolored inside the large object.
            if ra != rb or abs(ca - cb) != 1:
                continue

            left, right = sorted(cells, key=lambda rc: rc[1])
            target = int(x[left[0], left[1]])
            source = int(x[right[0], right[1]])

            if target == source:
                continue
            mapping[source] = target

        if not mapping:
            return None

        changed = False
        for source, target in mapping.items():
            mask = cropped == source
            if np.any(mask):
                cropped[mask] = target
                changed = True

        return cropped if changed else None

    # -------------------------
    # single 2 -> V plus inner diagonals
    # -------------------------
    def _expand_single_two_to_v_with_inner_diagonals(self, x):
        if x.ndim != 2 or x.shape[0] != 1:
            return None

        coords = np.argwhere(x == 2)
        nonzero = np.argwhere(x != 0)
        if len(coords) != 1 or len(nonzero) != 1:
            return None

        _, pivot = coords[0]
        n = x.shape[1]
        out = np.zeros((n, n), dtype=x.dtype)

        for r in range(n):
            left = pivot - r
            right = pivot + r
            if 0 <= left < n:
                out[r, left] = 2
            if 0 <= right < n:
                out[r, right] = 2

        m = 0
        while True:
            base_c = pivot - 1 - 2 * m
            start_c = max(0, base_c)
            start_r = 3 + 2 * m + max(0, -base_c)

            if start_r >= n:
                break

            r, c = start_r, start_c
            while r < n and c < n:
                out[r, c] = 1
                r += 1
                c += 1

            m += 1

        return out

    # -------------------------
    # draw between two 2s, turning crossed 1s into 3s
    # -------------------------
    def _draw_line_between_twos_with_one_to_three(self, x):
        coords = [tuple(int(v) for v in rc) for rc in np.argwhere(x == 2)]
        if len(coords) != 2:
            return None

        (r1, c1), (r2, c2) = coords
        dr = r2 - r1
        dc = c2 - c1

        if dr == 0:
            step_r, step_c = 0, 1 if dc > 0 else -1
            steps = abs(dc)
        elif dc == 0:
            step_r, step_c = 1 if dr > 0 else -1, 0
            steps = abs(dr)
        elif abs(dr) == abs(dc):
            step_r = 1 if dr > 0 else -1
            step_c = 1 if dc > 0 else -1
            steps = abs(dr)
        else:
            return None

        out = x.copy()
        for k in range(steps + 1):
            r = r1 + step_r * k
            c = c1 + step_c * k
            out[r, c] = 3 if x[r, c] == 1 else 2

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
