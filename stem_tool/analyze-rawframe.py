import os
import json
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import Normalize
from matplotlib import font_manager as fm


class GasTracker:
    def __init__(
        self,
        json_dir,
        image_path=None,
        raw_frame_dir=None,
        scale_csv=None,
        scale_value_nm=20.0,
        manual_scale_pixel_length=None,
        manual_nm_per_px=None,
        strict_scale_match=False,
        output_root=None,
        gas_category="gas",
        pin_category="pin"
    ):
        self.json_dir = json_dir
        self.image_path = image_path
        self.raw_frame_dir = str(raw_frame_dir) if raw_frame_dir else None
        self.image_dir = self.raw_frame_dir or (os.path.dirname(image_path) if image_path else None)
        self.scale_csv = scale_csv
        self.scale_value_nm = float(scale_value_nm)
        self.manual_scale_pixel_length = (
            None if manual_scale_pixel_length in (None, "") else float(manual_scale_pixel_length)
        )
        self.manual_nm_per_px = None if manual_nm_per_px in (None, "") else float(manual_nm_per_px)
        self.strict_scale_match = bool(strict_scale_match)
        self.gas_category = str(gas_category).strip() or "gas"
        self.pin_category = pin_category

        if self.manual_nm_per_px is not None and self.manual_nm_per_px <= 0:
            raise ValueError(f"manual_nm_per_px must be > 0, got {self.manual_nm_per_px}")
        if self.manual_scale_pixel_length is not None and self.manual_scale_pixel_length <= 0:
            raise ValueError(
                f"manual_scale_pixel_length must be > 0, got {self.manual_scale_pixel_length}"
            )
        if self.manual_nm_per_px is not None and self.manual_scale_pixel_length is not None:
            raise ValueError(
                "Specify either manual_nm_per_px or manual_scale_pixel_length, not both."
            )
        if self.manual_nm_per_px is None and self.manual_scale_pixel_length is not None:
            self.manual_nm_per_px = self.scale_value_nm / self.manual_scale_pixel_length

        self.output_root = self._resolve_output_root(output_root)
        os.makedirs(self.output_root, exist_ok=True)

        self.json_files = self._load_and_sort_jsons()

        # per-frame scale map: {frame_stem: nm_per_pixel}
        self.scale_map = {}
        self.fallback_nm_per_px = None
        self.max_nm_per_px = None
        self.min_nm_per_px = None
        self._warned_no_scale_csv = False
        self._warned_missing_scale_match = False
        if self.scale_csv is not None:
            self.scale_map = self._load_nm_per_px_map(self.scale_csv, default_scale_value_nm=self.scale_value_nm)
            if len(self.scale_map) > 0:
                vals = np.array(list(self.scale_map.values()), dtype=np.float64)
                self.fallback_nm_per_px = float(np.median(vals))
                self.max_nm_per_px = float(np.max(vals))
                self.min_nm_per_px = float(np.min(vals))
            else:
                raise ValueError(f"Scale CSV provided but no usable rows found: {self.scale_csv}")

        # 数据容器（全部使用真实尺寸：nm / nm^2）
        self.area_records = []        # [frame_id, frame_name, nm_per_px, area_nm2]
        self.contour_records = []     # [frame_id, frame_name, "(x_nm,y_nm)", ...]
        self.centroid_records = []    # [frame_id, frame_name, nm_per_px, cx_nm, cy_nm]
        self.object_records = []      # [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2]
        self.diameter_height_records = [] # [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, diameter_nm, height_nm]

        # pin 参考
        self.ref_pin_centroid = None
        self.last_shift = np.zeros(2)

        # 画图准备（优先使用 raw_frame_dir；image_path 仅作兼容兜底）
        self.W, self.H = None, None
        if self.image_path:
            img = Image.open(self.image_path)
            self.W, self.H = img.size
        elif self.image_dir:
            self.W, self.H = self._infer_frame_size(self.image_dir)

        # Make sure Chinese text can render on Windows (avoid "□□□" tofu boxes)
        self._configure_matplotlib_fonts()

    def _resolve_output_root(self, output_root):
        if output_root is None:
            return self.gas_category
        return str(output_root)

    def _ensure_output_root(self):
        os.makedirs(self.output_root, exist_ok=True)

    def _resolve_output_dir(self, output_dir, default_dir_name):
        if output_dir is None:
            return os.path.join(self.output_root, default_dir_name)
        if os.path.isabs(output_dir):
            return output_dir
        return os.path.join(self.output_root, output_dir)

    def _uses_pixel_units(self):
        return self.scale_csv is None and self.manual_nm_per_px is None

    def _length_unit(self):
        return "px" if self._uses_pixel_units() else "nm"

    def _area_unit(self):
        return f"{self._length_unit()}^2"

    def _speed_unit(self):
        return f"{self._length_unit()}/s"

    def _scale_column_name(self):
        return f"{self._length_unit()}_per_pixel"

    def _coord_column_name(self, axis_name):
        return f"{axis_name}_{self._length_unit()}"

    def _area_column_name(self, prefix="area"):
        return f"{prefix}_{self._length_unit()}2"

    def _distance_column_name(self, prefix):
        return f"{prefix}_{self._length_unit()}"

    def _contour_column_name(self):
        return f"contour_points_{self._length_unit()}"

    def _max_dist_label(self):
        return f"max_dist_{self._length_unit()}"

    def _infer_frame_size(self, image_dir):
        if not image_dir or not os.path.isdir(image_dir):
            return None, None

        possible_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        for frame_name in [Path(json_name).stem for json_name in self.json_files]:
            for ext in possible_exts:
                img_path = os.path.join(image_dir, frame_name + ext)
                if not os.path.exists(img_path):
                    continue
                try:
                    with Image.open(img_path) as img:
                        return img.size
                except Exception:
                    continue
        return None, None

    def _coords_nm_to_plot_px(self, coords_nm, frame_name=None, nm_per_px=None):
        coords_arr = np.asarray(coords_nm, dtype=np.float64)
        scale = nm_per_px
        if scale is None and frame_name is not None:
            try:
                scale = self._nm_per_px_for_frame(frame_name)
            except Exception:
                scale = None
        if scale is None:
            return coords_arr
        scale = float(scale)
        if abs(scale) <= 1e-12:
            return coords_arr
        return coords_arr / scale

    def _object_detections_by_frame(self):
        from collections import defaultdict

        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
            by_frame[int(frame_id)].append(
                (frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2))
            )
        return by_frame

    @staticmethod
    def _configure_matplotlib_fonts():
        """Configure Matplotlib fonts for Chinese text.

        If suitable CJK fonts aren't available, Matplotlib will fall back and may show tofu boxes.
        """
        preferred = [
            "Microsoft YaHei",  # 微软雅黑
            "SimHei",           # 黑体
            "PingFang SC",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

        try:
            available = {f.name for f in fm.fontManager.ttflist}
            chosen = [name for name in preferred if name in available]
            if chosen:
                plt.rcParams["font.sans-serif"] = chosen
        except Exception:
            # best-effort: still set a reasonable default list
            plt.rcParams["font.sans-serif"] = preferred

        # Global plotting font sizes: keep all generated chart text consistently larger.
        plt.rcParams["font.size"] = 17
        plt.rcParams["axes.titlesize"] = 20
        plt.rcParams["axes.labelsize"] = 17
        plt.rcParams["xtick.labelsize"] = 15
        plt.rcParams["ytick.labelsize"] = 15
        plt.rcParams["legend.fontsize"] = 14
        plt.rcParams["figure.titlesize"] = 20

        plt.rcParams["axes.unicode_minus"] = False

    @staticmethod
    def _parse_scale_value_to_nm(scale_value, unit):
        if scale_value is None:
            return None
        if unit is None:
            return float(scale_value)
        u = str(unit).strip().lower()
        v = float(scale_value)
        if u in {"nm", "nanometer", "nanometers"}:
            return v
        if u in {"um", "µm", "micrometer", "micrometers"}:
            return v * 1000.0
        if u in {"mm"}:
            return v * 1_000_000.0
        return v

    @classmethod
    def _load_nm_per_px_map(cls, csv_path, default_scale_value_nm=20.0):
        """Load per-image nm/px from a scalebar CSV.

        Supports:
        - minimal CSV: image,pixel_length
        - yolo_easyocr output: image,scale_value,unit,pixel_length,ratio,...

        Keying:
        - uses image basename stem, e.g. '..._000000000003'
        """
        csv_path = str(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Scale CSV not found: {csv_path}. "
                "Please provide a CSV with columns 'image' and 'pixel_length'."
            )

        nm_per_px = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = (row.get("image") or row.get("img") or "").strip()
                px_len = row.get("pixel_length")
                if img == "" or px_len in (None, ""):
                    continue

                try:
                    pixel_length = float(px_len)
                except Exception:
                    continue
                if pixel_length <= 0:
                    continue

                scale_value = row.get("scale_value")
                unit = row.get("unit")
                scale_nm = None
                if scale_value not in (None, ""):
                    try:
                        scale_nm = cls._parse_scale_value_to_nm(scale_value, unit)
                    except Exception:
                        scale_nm = None
                if scale_nm is None:
                    scale_nm = float(default_scale_value_nm)

                stem = Path(img).stem
                nm_per_px[stem] = float(scale_nm) / float(pixel_length)

        return nm_per_px

    def _nm_per_px_for_frame(self, frame_name):
        if self.scale_csv is None:
            if self.manual_nm_per_px is not None:
                return float(self.manual_nm_per_px)
            # No scale CSV: keep pipeline running in pixel-space (1 px = 1 px).
            if not self._warned_no_scale_csv:
                print(
                    "[warn] scale_csv is not set. Continue with fallback px_per_pixel=1.0 "
                    "(all exported values and plots will use pixel units)."
                )
                self._warned_no_scale_csv = True
            return 1.0
        v = self.scale_map.get(frame_name)
        if v is not None:
            return float(v)
        if self.strict_scale_match:
            raise KeyError(f"No scale entry for frame '{frame_name}' in {self.scale_csv}")
        # Non-strict mode: do not skip frame; use dataset-level fallback if available.
        if self.fallback_nm_per_px is not None:
            if not self._warned_missing_scale_match:
                print(
                    "[warn] Some frames have no matching scale in CSV. "
                    f"Using fallback median nm_per_px={self.fallback_nm_per_px:.6f}."
                )
                self._warned_missing_scale_match = True
            return float(self.fallback_nm_per_px)
        if not self._warned_missing_scale_match:
            print(
                "[warn] No matching scale and no fallback available. "
                "Use px_per_pixel=1.0 (pixel-scale units)."
            )
            self._warned_missing_scale_match = True
        return 1.0

    @staticmethod
    def _compute_droplet_dims_oriented(pts):
        """
        Compute droplet diameter/height from a fitted contact line on the original contour.

        The previous convex-hull heuristic often picked a chord on the dome rather than the
        actual contact line, which made the fitted rectangle drift. This version searches for
        the longest nearly straight contiguous contour segment that also acts as a supporting
        line for the rest of the droplet, then measures height as the farthest inward point.

        Returns:
            diameter (float)
            height (float)
            box_info (dict): {
                'baseline_p1': (x,y),
                'baseline_p2': (x,y),
                'apex_point': (x,y),
                'base_mid_point': (x,y),
                'corners': [(x,y), ...]
            }
        """
        if pts.shape[0] < 3:
            return 0.0, 0.0, None

        def fit_circle_kasa(x_vals, y_vals):
            """Kasa algebraic circle fit with proper data centering for numerical stability."""
            x = np.asarray(x_vals, dtype=np.float64)
            y = np.asarray(y_vals, dtype=np.float64)
            n = len(x)
            if n < 3:
                return None
            xm, ym = x.mean(), y.mean()
            u = x - xm
            v = y - ym
            Suu = np.dot(u, u)
            Svv = np.dot(v, v)
            Suv = np.dot(u, v)
            A = np.array([[Suu, Suv], [Suv, Svv]])
            b_vec = np.array([
                0.5 * (np.dot(u, u * u) + np.dot(u, v * v)),
                0.5 * (np.dot(v, v * v) + np.dot(v, u * u)),
            ])
            try:
                uc, vc = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                return None
            cx = xm + uc
            cy = ym + vc
            r_sq = uc * uc + vc * vc + (Suu + Svv) / n
            if r_sq <= 0:
                return None
            r = float(np.sqrt(r_sq))
            resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
            rms = float(np.sqrt(np.mean(resid ** 2))) if resid.size > 0 else 0.0
            return float(cx), float(cy), r, rms

        def fit_spherical_cap_1d(x_dome, y_dome, half_span):
            """
            Constrained spherical-cap fit for a sessile droplet.

            The center is fixed on the perpendicular bisector of the contact span
            (cx = 0 in local frame), so only cy is optimised.  The contact radius
            a = half_span fixes the sphere radius once cy is known:
                R = sqrt(a^2 + cy^2)
            Height of the cap above the baseline:
                h = cy + R

            This 1-D optimisation is much more robust than an unconstrained 3-D
            algebraic fit and is the physically correct model for a sessile droplet.

            Returns (cy, radius, rms) or None.
            """
            from scipy.optimize import minimize_scalar
            x = np.asarray(x_dome, dtype=np.float64)
            y = np.asarray(y_dome, dtype=np.float64)
            if len(x) < 3 or half_span <= 0:
                return None

            def cost(cy_val):
                r_val = np.sqrt(half_span * half_span + cy_val * cy_val)
                dists = np.sqrt(x * x + (y - cy_val) ** 2)
                return float(np.sum((dists - r_val) ** 2))

            # Initial estimate from apex height via spherical-cap geometry:
            # h = cy + R, R^2 = a^2 + cy^2  =>  cy = (h^2 - a^2) / (2*h)
            h_est = float(np.max(y)) if y.size > 0 else half_span
            a = half_span
            cy_init = (h_est * h_est - a * a) / (2.0 * h_est) if h_est > 1e-6 else 0.0
            lo = cy_init - 2.0 * a
            hi = cy_init + 2.0 * a

            try:
                opt = minimize_scalar(cost, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-4, 'maxiter': 300})
                cy = float(opt.x)
            except Exception:
                cy = cy_init

            r = float(np.sqrt(a * a + cy * cy))
            resid = np.sqrt(x * x + (y - cy) ** 2) - r
            rms = float(np.sqrt(np.mean(resid ** 2))) if resid.size > 0 else 0.0
            return cy, r, rms

        pts = np.asarray(pts, dtype=np.float64)
        n_pts = int(pts.shape[0])
        if n_pts < 3:
            return 0.0, 0.0, None

        bbox_min = np.min(pts, axis=0)
        bbox_max = np.max(pts, axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        if diag <= 1e-6:
            return 0.0, 0.0, None

        # ---- Convex-hull chord sweep for contact baseline ----
        #
        # The contact line of a sessile droplet is the longest chord that:
        #   (a) acts as an approximate supporting line (all points on one side), and
        #   (b) yields H <= D (physical constraint for any spherical cap).
        #
        # We iterate over all pairs of convex-hull vertices.  A typical hull has
        # 8-20 vertices, so this is O(n_hull^2) ~ a few hundred iterations at most.
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            hull_verts = pts[hull.vertices]
        except Exception:
            hull_verts = pts

        n_hull = len(hull_verts)
        outside_tol = max(2.0, diag * 0.025)
        min_chord = max(8.0, diag * 0.15)

        best = None

        for i in range(n_hull):
            for j in range(n_hull):
                if i == j:
                    continue
                p1 = hull_verts[i]
                p2 = hull_verts[j]
                chord = p2 - p1
                chord_len = float(np.linalg.norm(chord))
                if chord_len < min_chord:
                    continue

                direction = chord / chord_len
                normal = np.array([-direction[1], direction[0]], dtype=np.float64)

                # Signed distances of all contour pts from the line through p1
                signed = np.dot(pts - p1, normal)

                # Orient normal so that the majority of pts are on the positive side
                if np.sum(signed > 0) < np.sum(signed < 0):
                    normal = -normal
                    signed = -signed

                # Reject if too many points lie on the wrong side of the line
                outside_frac = float(np.mean(signed < -outside_tol))
                if outside_frac > 0.05:
                    continue

                height = float(np.max(signed))
                if height <= 1.0:
                    continue

                # Score = chord length, with strong penalty when H > D
                hd = height / chord_len
                score = chord_len
                if hd > 1.0:
                    score *= 1.0 / (1.0 + (hd - 1.0) ** 2 * 50.0)

                if best is None or score > best["score"] + 1e-9:
                    best = {
                        "score": score,
                        "p1": p1, "p2": p2,
                        "direction": direction, "normal": normal,
                        "signed": signed,
                        "diameter": chord_len, "height": height,
                    }

        # ---- Fallback: PCA orientation ----
        if best is None:
            centroid = np.mean(pts, axis=0)
            try:
                _u, _s, vh = np.linalg.svd(pts - centroid, full_matrices=False)
                direction = np.asarray(vh[0], dtype=np.float64)
                direction /= max(float(np.linalg.norm(direction)), 1e-9)
            except Exception:
                direction = np.array([1.0, 0.0], dtype=np.float64)
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            signed = np.dot(pts - centroid, normal)
            if np.sum(signed > 0) < np.sum(signed < 0):
                normal = -normal
                signed = -signed
            u_all = np.dot(pts - centroid, direction)
            base_off = float(np.min(signed))
            signed -= base_off
            diameter = float(np.max(u_all) - np.min(u_all))
            height = float(np.max(signed))
            apex_world = pts[int(np.argmax(signed))]
            base_p1_world = centroid + direction * float(np.min(u_all)) + normal * base_off
            base_p2_world = centroid + direction * float(np.max(u_all)) + normal * base_off
            base_mid_world = 0.5 * (base_p1_world + base_p2_world)
            c1, c2 = base_p1_world, base_p2_world
            c3, c4 = c2 + normal * height, c1 + normal * height
            return diameter, height, {
                "baseline_p1": c1, "baseline_p2": c2,
                "apex_point": apex_world, "base_mid_point": base_mid_world,
                "corners": [c1, c2, c3, c4],
                "arc_points": None, "fit_center": None, "fit_radius": None,
            }

        # ---- Extract baseline from best hull chord ----
        base_p1_world = best["p1"]
        base_p2_world = best["p2"]
        direction = best["direction"]
        normal = best["normal"]
        signed_all = best["signed"]
        diameter = best["diameter"]
        height = best["height"]

        apex_idx = int(np.argmax(signed_all))
        apex_world = pts[apex_idx]
        base_mid_world = 0.5 * (base_p1_world + base_p2_world)

        c1, c2 = base_p1_world, base_p2_world
        c3, c4 = c2 + normal * height, c1 + normal * height

        # ---- Spherical-cap refinement on dome points ----
        fit_center_world = None
        fit_radius = None

        baseline_mid = base_mid_world
        local_x = np.dot(pts - baseline_mid, direction)
        local_y = np.dot(pts - baseline_mid, normal)
        dome_mask = local_y > max(0.5, height * 0.03)
        half_span = diameter / 2.0

        if int(np.sum(dome_mask)) >= 4 and half_span > 1e-6:
            x_dome = local_x[dome_mask]
            y_dome = local_y[dome_mask]

            # Primary: constrained spherical-cap (cx = 0)
            cap_fit = fit_spherical_cap_1d(x_dome, y_dome, half_span)
            used_constrained = False
            if cap_fit is not None:
                fit_cy, fit_r, fit_rms = cap_fit
                fit_height = float(fit_cy + fit_r)
                if fit_height > 1.0 and fit_rms <= max(5.0, height * 0.30):
                    height = float(fit_height)
                    apex_world = baseline_mid + normal * fit_height
                    c3 = c2 + normal * height
                    c4 = c1 + normal * height
                    fit_center_world = baseline_mid + normal * fit_cy
                    fit_radius = float(fit_r)
                    used_constrained = True

            # Fallback: Kasa unconstrained
            if not used_constrained:
                kasa = fit_circle_kasa(x_dome, y_dome)
                if kasa is not None:
                    fit_cx, fit_cy, fit_r, fit_rms = kasa
                    discriminant = fit_r * fit_r - fit_cy * fit_cy
                    if discriminant > 1.0 and fit_rms <= max(5.0, height * 0.30):
                        half_span_fit = float(np.sqrt(discriminant))
                        fit_height = float(fit_cy + fit_r)
                        if fit_height > 1.0:
                            diameter = float(2.0 * half_span_fit)
                            height = float(fit_height)
                            base_p1_world = baseline_mid + direction * float(fit_cx - half_span_fit)
                            base_p2_world = baseline_mid + direction * float(fit_cx + half_span_fit)
                            base_mid_world = baseline_mid + direction * float(fit_cx)
                            apex_world = base_mid_world + normal * fit_height
                            c1, c2 = base_p1_world, base_p2_world
                            c3, c4 = c2 + normal * height, c1 + normal * height
                            fit_center_world = baseline_mid + direction * float(fit_cx) + normal * float(fit_cy)
                            fit_radius = float(fit_r)

        return diameter, height, {
            "baseline_p1": base_p1_world,
            "baseline_p2": base_p2_world,
            "apex_point": apex_world,
            "base_mid_point": base_mid_world,
            "corners": [c1, c2, c3, c4],
            "arc_points": None,
            "fit_center": fit_center_world,
            "fit_radius": fit_radius,
        }

    # -----------------------------
    # 工具函数
    # -----------------------------
    def _load_and_sort_jsons(self):
        import re

        files = [
            f for f in os.listdir(self.json_dir)
            if f.endswith(".json")
        ]

        def _sort_key(name):
            stem = Path(name).stem
            m = re.search(r"(\d+)$", stem)
            if m is not None:
                # Keep original behavior for names ending with numeric frame id.
                return (0, int(m.group(1)), stem.lower())
            # Fallback: non-numeric names are sorted lexicographically after numeric ones.
            return (1, 0, stem.lower())

        files.sort(key=_sort_key)
        return files

    @staticmethod
    def polygon_area(coords):
        """
        coords: (N,2) 不需要闭合
        """
        x = coords[:, 0]
        y = coords[:, 1]
        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) -
            np.dot(y, np.roll(x, -1))
        )

    # -----------------------------
    # 主处理流程
    # -----------------------------
    def process_all_frames(self):
        for frame_id, json_name in enumerate(self.json_files):
            json_path = os.path.join(self.json_dir, json_name)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            frame_name = Path(json_name).stem
            try:
                nm_per_px = self._nm_per_px_for_frame(frame_name)
            except Exception as e:
                print(f"[skip] frame_id={frame_id} frame_name={frame_name}: scale lookup error: {e}")
                continue
            if nm_per_px is None:
                print(f"[skip] frame_id={frame_id} frame_name={frame_name}: no matching scale in CSV")
                continue

            shift = self._compute_pin_shift(data)

            self._process_gas_objects(
                data,
                frame_id,
                frame_name,
                nm_per_px,
                shift
            )

    def _compute_pin_shift(self, data):
        pin_pts = []
        for obj in data.get("objects", []):
            if obj.get("category") == self.pin_category:
                pin_pts.append(
                    np.array(obj["segmentation"], dtype=np.float32)
                )

        if len(pin_pts) > 0:
            pin_pts = np.vstack(pin_pts)
            pin_centroid = pin_pts.mean(axis=0)

            if self.ref_pin_centroid is None:
                self.ref_pin_centroid = pin_centroid.copy()

            shift = pin_centroid - self.ref_pin_centroid
            self.last_shift = shift
        else:
            shift = self.last_shift

        return shift

    def _process_gas_objects(self, data, frame_id, frame_name, nm_per_px, shift):
        for obj in data.get("objects", []):
            if obj.get("category") != self.gas_category:
                continue

            pts = np.array(obj["segmentation"], dtype=np.float32)
            pts = pts - shift   # ★ 去整体漂移

            if pts.shape[0] < 3:
                continue

            # ---- 面积 ----
            area_px2 = self.polygon_area(pts)
            area_nm2 = float(area_px2) * float(nm_per_px) * float(nm_per_px)
            self.area_records.append([frame_id, frame_name, float(nm_per_px), area_nm2])

            # ---- 质心 ----
            centroid = pts.mean(axis=0)
            cx_px, cy_px = float(centroid[0]), float(centroid[1])
            cx_nm, cy_nm = cx_px * float(nm_per_px), cy_px * float(nm_per_px)
            self.centroid_records.append([frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm])

            # ---- Diameter and Height (Rotating Calipers / Minimum Area Rectangle) ----
            # The droplet is a semi-circle projected essentially as a "D" shape.
            # The "bottom" is the flat side of the D. 
            # We need to find the orientation of this flat side to measure Diameter (length of flat side)
            # and Height (max perpendicular distance from flat side).
            
            # Use Rotating Calipers via Minimum Area Rectangle to find the major axes.
            # For a semi-circle, the Minimum Area Rectangle usually aligns such that one side is the diameter.
            
            from scipy.spatial import ConvexHull
            


            try:
                # Use the new robust method
                if len(pts) >= 3:
                     # Use the new robust method
                    d_px, h_px, box_info = self._compute_droplet_dims_oriented(pts)
                    d_nm = d_px * nm_per_px
                    h_nm = h_px * nm_per_px
                    
                    self.diameter_height_records.append([
                        frame_id, frame_name, nm_per_px, cx_nm, cy_nm, d_nm, h_nm, box_info
                    ])
                else:
                    self.diameter_height_records.append([frame_id, frame_name, nm_per_px, cx_nm, cy_nm, 0, 0, {}])
            except Exception as e:
                # Fallback to AABB
                print(f"Error in oriented calc: {e}, using AABB")
                min_x, min_y = pts.min(axis=0)
                max_x, max_y = pts.max(axis=0)
                d_nm = (max_x - min_x) * nm_per_px
                h_nm = (max_y - min_y) * nm_per_px
                # Dummy values for the rest
                self.diameter_height_records.append([frame_id, frame_name, nm_per_px, cx_nm, cy_nm, d_nm, h_nm, {}])

            # ---- 每个目标的聚合记录（用于追踪面积曲线）----
            self.object_records.append([frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm, area_nm2])

            # ---- 轮廓（每帧一行）----
            row = [frame_id, frame_name]
            for (x, y) in pts:
                x_nm = float(x) * float(nm_per_px)
                y_nm = float(y) * float(nm_per_px)
                row.append(f"({x_nm:.3f},{y_nm:.3f})")
            self.contour_records.append(row)

    # -----------------------------
    # 数据导出
    # -----------------------------
    def _build_export_instance_ids(self, max_dist=50.0, id_mode="event", use_display_id=True):
        """Build per-record droplet ids aligned with object_records order."""
        if len(self.object_records) == 0:
            return []

        from collections import defaultdict

        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
            by_frame[int(frame_id)].append((frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2)))

        mode = str(id_mode).strip().lower()
        if mode != "event":
            raise NotImplementedError("export_results currently supports id_mode='event' only")

        series_by_id, assigned_ids_by_frame, _events = self._build_event_id_series_with_assignments(
            by_frame,
            max_dist=max_dist,
        )

        if bool(use_display_id):
            display_id_of = self._display_id_mapping(series_by_id)
            assigned_ids_by_frame = {
                int(frame_id): [int(display_id_of.get(int(instance_id), int(instance_id))) for instance_id in ids]
                for frame_id, ids in assigned_ids_by_frame.items()
            }

        export_ids = []
        for frame_id in sorted(assigned_ids_by_frame.keys()):
            export_ids.extend(assigned_ids_by_frame[frame_id])

        if len(export_ids) != len(self.object_records):
            raise ValueError(
                f"Export instance-id count mismatch: ids={len(export_ids)} object_records={len(self.object_records)}"
            )

        return export_ids

    def export_results(self, max_dist=50.0, id_mode="event", use_display_id=True):
        self._ensure_output_root()
        export_ids = self._build_export_instance_ids(
            max_dist=max_dist,
            id_mode=id_mode,
            use_display_id=use_display_id,
        )

        scale_col = self._scale_column_name()
        area_col = self._area_column_name()
        contour_col = self._contour_column_name()
        cx_col = self._coord_column_name("cx")
        cy_col = self._coord_column_name("cy")
        diameter_col = self._distance_column_name("diameter")
        height_col = self._distance_column_name("height")

        # 面积
        path1 = os.path.join(self.output_root, f"{self.gas_category}_area_vs_frame.csv")
        with open(path1, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", scale_col, area_col])
            writer.writerows(
                [[int(instance_id), frame_id, frame_name, f"{nm_per_px:.6f}", f"{area_nm2:.6f}"]
                 for instance_id, (frame_id, frame_name, nm_per_px, area_nm2) in zip(export_ids, self.area_records)]
            )

        # 轮廓（每帧一行）
        path2 = os.path.join(self.output_root, f"{self.gas_category}_contours_by_frame.csv")
        with open(path2, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", contour_col])
            writer.writerows(
                [[int(instance_id)] + row for instance_id, row in zip(export_ids, self.contour_records)]
            )

        # 质心
        path3 = os.path.join(self.output_root, f"{self.gas_category}_centroids.csv")
        with open(path3, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", scale_col, cx_col, cy_col])
            writer.writerows(
                [[int(instance_id), frame_id, frame_name, f"{nm_per_px:.6f}", f"{cx_nm:.6f}", f"{cy_nm:.6f}"]
                 for instance_id, (frame_id, frame_name, nm_per_px, cx_nm, cy_nm) in zip(export_ids, self.centroid_records)]
            )

        # Diameter and Height
        path4 = os.path.join(self.output_root, f"{self.gas_category}_diameter_height_vs_frame.csv")
        with open(path4, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", scale_col, cx_col, cy_col, diameter_col, height_col])
            for instance_id, row in zip(export_ids, self.diameter_height_records):
                # row structure: [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, d_nm, h_nm, min_x, min_y, max_x, max_y]
                # we only export the first 7 fields here
                writer.writerow([int(instance_id), row[0], row[1], f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}", f"{row[5]:.6f}", f"{row[6]:.6f}"])

        print("Export finished:")
        print(f" - {path1}")
        print(f" - {path2}")
        print(f" - {path3}")
        print(f" - {path4}")

    def annotate_images(
        self,
        output_dir=None,
        label_ids=False,
        id_mode="event",
        max_dist=50.0,
        min_track_length=0,
        use_display_id=True,
    ):
        output_dir = self._resolve_output_dir(output_dir, "annotated_images")
                 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not self.image_dir:
            print("[skip] annotate_images: no image directory is available. Set raw_frame_dir or image_path.")
            return
        
        print(f"Annotating images to {output_dir}...")
        
        try:
            # Try to start with a slightly larger font if possible
            font = ImageFont.truetype("arial.ttf", 38)
        except OSError:
            font = ImageFont.load_default()

        assigned_ids_by_frame = None
        display_id_of = None
        if bool(label_ids):
            if len(self.object_records) == 0:
                print("[warn] label_ids=True but object_records is empty; run process_all_frames() first.")
            else:
                from collections import defaultdict

                detections_by_frame = defaultdict(list)
                for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
                    detections_by_frame[int(frame_id)].append(
                        (str(frame_name), float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2))
                    )

                mode = str(id_mode).strip().lower()
                if mode != "event":
                    raise NotImplementedError("annotate_images(label_ids=True) currently supports id_mode='event' only")

                series_by_id, assigned_ids_by_frame, _events = self._build_event_id_series_with_assignments(
                    detections_by_frame, max_dist=max_dist
                )

                series_by_id_for_display = {
                    k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)
                }
                if bool(use_display_id):
                    display_id_of = self._display_id_mapping(series_by_id_for_display)
                else:
                    display_id_of = None

                print(
                    f"Annotate IDs enabled: mode={mode}, {self._max_dist_label()}={float(max_dist)}, "
                    f"min_track_length={int(min_track_length)}, use_display_id={bool(use_display_id)}, "
                    f"ids_total={len(series_by_id)}"
                )

        # For robust annotation across categories:
        # - Draw the segmentation contours for self.gas_category.
        # - Only for nanodroplet, additionally draw diameter/height overlays.
        for frame_id, json_name in enumerate(self.json_files):
            frame_name = Path(json_name).stem
            # Find image
            img_path = None
            possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
            for ext in possible_exts:
                p = os.path.join(self.image_dir, frame_name + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            if not img_path:
                continue

            try:
                with Image.open(img_path) as im:
                    img_out = im.convert("RGB")
                    draw = ImageDraw.Draw(img_out)

                    json_path = os.path.join(self.json_dir, json_name)
                    with open(json_path, "r", encoding="utf-8") as f:
                        jdata = json.load(f)

                    try:
                        nm_per_px = self._nm_per_px_for_frame(frame_name)
                    except Exception:
                        nm_per_px = None

                    ids_this_frame = None
                    if assigned_ids_by_frame is not None:
                        ids_this_frame = assigned_ids_by_frame.get(int(frame_id))

                    obj_idx = 0

                    for obj in jdata.get("objects", []):
                        if obj.get("category") != self.gas_category:
                            continue
                        pts_raw = np.array(obj.get("segmentation", []), dtype=np.float32)
                        if pts_raw.shape[0] < 3:
                            obj_idx += 1
                            continue

                        # ID label (use the same within-frame order as JSON/category iteration)
                        if bool(label_ids) and ids_this_frame is not None and obj_idx < len(ids_this_frame):
                            try:
                                instance_id = int(ids_this_frame[obj_idx])
                                if display_id_of is not None:
                                    disp = int(display_id_of.get(instance_id, 0))
                                    id_text = str(disp) if disp > 0 else str(instance_id)
                                else:
                                    id_text = str(instance_id)

                                cx_px = float(np.mean(pts_raw[:, 0]))
                                cy_px = float(np.mean(pts_raw[:, 1]))
                                r = 6
                                draw.ellipse((cx_px - r, cy_px - r, cx_px + r, cy_px + r), outline="orange", width=3)
                                draw.text(
                                    (cx_px - 20, cy_px -18),
                                    id_text,
                                    fill="orange",
                                    font=font,
                                    stroke_width=2,
                                    stroke_fill="black",
                                )
                            except Exception:
                                pass

                        # draw contour
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        draw.polygon(poly, outline="lime", width=2)

                        # droplet-only: draw diameter/height overlay
                        if str(self.gas_category).lower() == "nanodroplet" and nm_per_px is not None:
                            try:
                                d_px, h_px, box_info = self._compute_droplet_dims_oriented(pts_raw)
                                d_nm = float(d_px) * float(nm_per_px)
                                h_nm = float(h_px) * float(nm_per_px)

                                corners = box_info.get("corners")
                                if corners is not None:
                                    corners_arr = np.array(corners, dtype=np.float32)
                                    rect_poly = [tuple(map(float, p)) for p in corners_arr]
                                    draw.polygon(rect_poly, outline="cyan", width=2)

                                    text = f"D:{d_nm:.1f} {self._length_unit()}\nH:{h_nm:.1f} {self._length_unit()}"
                                    cx = float(corners_arr[:, 0].mean())
                                    cy = float(corners_arr[:, 1].mean())
                                    draw.text((cx, cy), text, fill="yellow", font=font)

                                baseline_p1 = box_info.get("baseline_p1")
                                baseline_p2 = box_info.get("baseline_p2")
                                if baseline_p1 is not None and baseline_p2 is not None:
                                    draw.line([tuple(map(float, baseline_p1)), tuple(map(float, baseline_p2))], fill="red", width=3)

                                apex = box_info.get("apex_point")
                                base_mid = box_info.get("base_mid_point")
                                if apex is not None and base_mid is not None:
                                    draw.line([tuple(map(float, apex)), tuple(map(float, base_mid))], fill="magenta", width=2)
                            except Exception:
                                # best-effort; keep contour even if dims fail
                                pass

                        obj_idx += 1

                    out_path = os.path.join(output_dir, frame_name + ".png")
                    img_out.save(out_path)

            except Exception as e:
                print(f"Error annotating {frame_name}: {e}")

    def annotate_images_on_rawframe(
        self,
        raw_frame_dir,
        output_dir=None,
        label_ids=False,
        id_mode="event",
        max_dist=50.0,
        min_track_length=3,
        use_display_id=True,
        mask_alpha=120,
    ):
        """Generate annotated images using the original raw frames as background.

        Semi-transparent filled masks are drawn for each detected object, plus
        contour outlines and analysis annotations (IDs, diameter/height).

        Args:
            raw_frame_dir: Directory containing the original raw frame images.
                           Frame filenames must match JSON stem names.
            output_dir: Where to save results. Defaults to
                        <output_root>/annotated_rawframe. Relative paths are
                        joined with output_root.
            label_ids: Whether to draw instance ID labels.
            id_mode: Tracking mode – only 'event' is supported.
            max_dist: Maximum linking distance in nm for ID assignment.
            min_track_length: Minimum number of frames a track must span to
                              receive a display ID label.
            use_display_id: Remap internal IDs to compact 1-based display IDs.
            mask_alpha: Alpha value (0-255) for filled mask overlays.
        """
        output_dir = self._resolve_output_dir(output_dir, "annotated_rawframe")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Annotating raw-frame images to {output_dir}...")

        # Font is initialised per-frame based on image size to avoid oversized labels.

        # Category-based fixed colour: nanocluster=red, nanodroplet=blue, gas=green, others=orange
        _CATEGORY_COLOR = {
            "nanocluster": (220, 30, 30),
            "nanodroplet": (30, 100, 255),
            "gas":         (0, 200, 80),
        }
        _cat_rgb = _CATEGORY_COLOR.get(str(self.gas_category).lower(), (255, 140, 0))
        _fill_rgba = _cat_rgb + (mask_alpha,)
        _outline_rgb = _cat_rgb

        # ---- Build ID assignments (same logic as annotate_images) ----
        assigned_ids_by_frame = None
        display_id_of = None
        allowed_instance_ids = None
        if bool(label_ids):
            if len(self.object_records) == 0:
                print("[warn] label_ids=True but object_records is empty; run process_all_frames() first.")
            else:
                from collections import defaultdict

                detections_by_frame = defaultdict(list)
                for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
                    detections_by_frame[int(frame_id)].append(
                        (str(frame_name), float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2))
                    )

                mode = str(id_mode).strip().lower()
                if mode != "event":
                    raise NotImplementedError(
                        "annotate_images_on_rawframe(label_ids=True) supports id_mode='event' only"
                    )

                series_by_id, assigned_ids_by_frame, _events = self._build_event_id_series_with_assignments(
                    detections_by_frame, max_dist=max_dist
                )

                series_by_id_for_display = {
                    k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)
                }
                allowed_instance_ids = set(int(k) for k in series_by_id_for_display.keys())
                if bool(use_display_id):
                    display_id_of = self._display_id_mapping(series_by_id_for_display)
                else:
                    display_id_of = None

                print(
                    f"Annotate IDs enabled: mode={mode}, {self._max_dist_label()}={float(max_dist)}, "
                    f"min_track_length={int(min_track_length)}, use_display_id={bool(use_display_id)}, "
                    f"ids_total={len(series_by_id)}"
                )

        # ---- Per-frame annotation ----
        possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

        for frame_id, json_name in enumerate(self.json_files):
            frame_name = Path(json_name).stem

            # Locate raw frame image
            raw_img_path = None
            for ext in possible_exts:
                p = os.path.join(raw_frame_dir, frame_name + ext)
                if os.path.exists(p):
                    raw_img_path = p
                    break
            if not raw_img_path:
                continue

            try:
                with Image.open(raw_img_path) as raw_im:
                    # Convert to RGBA so we can composite a mask layer
                    bg = raw_im.convert("RGBA")
                    W, H = bg.size

                    # Adaptive text size for different frame resolutions.
                    # Example: 512px frame -> ~14px font; 1024px frame -> ~24px font.
                    font_px = max(18, min(30, int(round(min(W, H) * 0.028))))
                    try:
                        font = ImageFont.truetype("arial.ttf", font_px)
                    except OSError:
                        font = ImageFont.load_default()
                    id_offset_x = int(max(8, round(font_px * 0.8)))
                    id_offset_y = int(max(8, round(font_px * 0.7)))
                    centroid_r = int(max(3, round(font_px * 0.22)))
                    stroke_w = int(max(1, round(font_px * 0.12)))

                    # Transparent overlay for filled masks
                    mask_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_layer)

                    json_path = os.path.join(self.json_dir, json_name)
                    with open(json_path, "r", encoding="utf-8") as f:
                        jdata = json.load(f)

                    try:
                        nm_per_px = self._nm_per_px_for_frame(frame_name)
                    except Exception:
                        nm_per_px = None

                    ids_this_frame = None
                    if assigned_ids_by_frame is not None:
                        ids_this_frame = assigned_ids_by_frame.get(int(frame_id))

                    obj_idx = 0
                    objects_this_frame = [
                        obj for obj in jdata.get("objects", [])
                        if obj.get("category") == self.gas_category
                    ]

                    for obj in objects_this_frame:
                        pts_raw = np.array(obj.get("segmentation", []), dtype=np.float32)
                        if pts_raw.shape[0] < 3:
                            obj_idx += 1
                            continue

                        # Draw filled semi-transparent mask (uniform colour per category)
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        mask_draw.polygon(poly, fill=_fill_rgba, outline=None)

                        obj_idx += 1

                    # Composite mask layer onto background
                    img_out = Image.alpha_composite(bg, mask_layer).convert("RGB")
                    draw = ImageDraw.Draw(img_out)

                    # Second pass: outlines + text on top of composited image
                    obj_idx = 0
                    for obj in objects_this_frame:
                        pts_raw = np.array(obj.get("segmentation", []), dtype=np.float32)
                        if pts_raw.shape[0] < 3:
                            obj_idx += 1
                            continue

                        # Draw contour outline
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        draw.polygon(poly, outline=_outline_rgb, width=2)

                        # ID label
                        if bool(label_ids) and ids_this_frame is not None and obj_idx < len(ids_this_frame):
                            try:
                                instance_id = int(ids_this_frame[obj_idx])
                                draw_id_label = True

                                # Hide short-lived/noisy tracks to reduce excessive labels.
                                if allowed_instance_ids is not None and instance_id not in allowed_instance_ids:
                                    draw_id_label = False

                                if display_id_of is not None:
                                    disp = int(display_id_of.get(instance_id, 0))
                                    if disp <= 0:
                                        draw_id_label = False
                                    id_text = str(disp) if disp > 0 else ""
                                else:
                                    id_text = str(instance_id)

                                if draw_id_label:
                                    cx_px = float(np.mean(pts_raw[:, 0]))
                                    cy_px = float(np.mean(pts_raw[:, 1]))
                                    r = centroid_r
                                    draw.ellipse(
                                        (cx_px - r, cy_px - r, cx_px + r, cy_px + r),
                                        outline=_outline_rgb,
                                        width=2,
                                    )
                                    draw.text(
                                        (cx_px - id_offset_x, cy_px - id_offset_y),
                                        id_text,
                                        fill="orange",
                                        font=font,
                                        stroke_width=stroke_w,
                                        stroke_fill="black",
                                    )
                            except Exception:
                                pass

                        # Diameter / height overlay (nanodroplet only)
                        if str(self.gas_category).lower() == "nanodroplet" and nm_per_px is not None:
                            try:
                                d_px, h_px, box_info = self._compute_droplet_dims_oriented(pts_raw)
                                d_nm = float(d_px) * float(nm_per_px)
                                h_nm = float(h_px) * float(nm_per_px)

                                corners = box_info.get("corners")
                                if corners is not None:
                                    corners_arr = np.array(corners, dtype=np.float32)
                                    rect_poly = [tuple(map(float, p)) for p in corners_arr]
                                    draw.polygon(rect_poly, outline="cyan", width=2)
                                    text = f"D:{d_nm:.1f} {self._length_unit()}\nH:{h_nm:.1f} {self._length_unit()}"
                                    cx = float(corners_arr[:, 0].mean())
                                    cy = float(corners_arr[:, 1].mean())
                                    draw.text((cx, cy), text, fill="yellow", font=font)

                                baseline_p1 = box_info.get("baseline_p1")
                                baseline_p2 = box_info.get("baseline_p2")
                                if baseline_p1 is not None and baseline_p2 is not None:
                                    draw.line(
                                        [tuple(map(float, baseline_p1)), tuple(map(float, baseline_p2))],
                                        fill="red",
                                        width=3,
                                    )

                                apex = box_info.get("apex_point")
                                base_mid = box_info.get("base_mid_point")
                                if apex is not None and base_mid is not None:
                                    draw.line(
                                        [tuple(map(float, apex)), tuple(map(float, base_mid))],
                                        fill="magenta",
                                        width=2,
                                    )
                            except Exception:
                                pass

                        obj_idx += 1

                    out_path = os.path.join(output_dir, frame_name + ".png")
                    img_out.save(out_path)

            except Exception as e:
                print(f"Error annotating raw frame {frame_name}: {e}")

        print(f"Raw-frame annotation complete: {output_dir}")

    def annotate_allcategories_on_rawframe(
        self,
        raw_frame_dir,
        output_dir=None,
        mask_alpha=120,
        show_centroid=True,
        label_ids=True,
        max_dist=50.0,
        use_display_id=True,
    ):
        """Generate annotated images using the original raw frames as background,
        drawing masks and outlines for ALL annotation categories in each JSON.

        Category colour mapping:
            nanocluster -> red  (220, 30, 30)
            nanodroplet -> blue (30, 100, 255)
            gas         -> green (0, 200, 80)
            pin         -> yellow (240, 200, 0)  (skipped)
            others      -> orange (255, 140, 0)

        Instance IDs are tracked per category and match the exported CSV tables
        for self.gas_category.  Other categories receive their own consistent IDs.

        Args:
            raw_frame_dir: Directory containing the original raw frame images.
            output_dir: Output directory. Defaults to
                        <output_root>/annotated_allcat_rawframe.
            mask_alpha: Alpha value (0-255) for filled mask overlays.
            show_centroid: Whether to draw a centroid dot on each instance.
            label_ids: Whether to draw instance ID labels.
            max_dist: Maximum centroid linking distance (nm) for ID tracking.
            use_display_id: Remap internal IDs to compact 1-based display IDs.
        """
        output_dir = self._resolve_output_dir(output_dir, "annotated_allcat_rawframe")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Annotating all-categories raw-frame images to {output_dir}...")

        # Font is initialised per-frame based on image size to avoid oversized labels.

        _CATEGORY_COLOR = {
            "nanocluster": (220, 30, 30),
            "nanodroplet": (30, 100, 255),
            "gas":         (0, 200, 80),
            "pin":         (240, 200, 0),
        }
        _DEFAULT_COLOR = (255, 140, 0)

        # ---- Pre-scan: build per-category detections for ID tracking ----
        from collections import defaultdict

        # cat_dets[cat][frame_id] = [(frame_name, nm_per_px, cx_nm, cy_nm, area_nm2), ...]
        cat_dets = defaultdict(lambda: defaultdict(list))

        # --- gas_category: use self.object_records (drift-corrected, same as CSV export) ---
        _gas_cat = str(self.gas_category).strip().lower()
        if self.object_records:
            for frame_id_r, frame_name_r, nm_per_px_r, cx_nm_r, cy_nm_r, area_nm2_r in self.object_records:
                cat_dets[_gas_cat][int(frame_id_r)].append(
                    (str(frame_name_r), float(nm_per_px_r), float(cx_nm_r), float(cy_nm_r), float(area_nm2_r))
                )

        # --- other categories: read from JSON (no drift correction available) ---
        for frame_id, json_name in enumerate(self.json_files):
            frame_name = Path(json_name).stem
            json_path = os.path.join(self.json_dir, json_name)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    jdata_pre = json.load(f)
            except Exception:
                continue
            try:
                nm_pre = self._nm_per_px_for_frame(frame_name)
            except Exception:
                nm_pre = None
            nm_pre_f = float(nm_pre) if nm_pre is not None else 1.0

            for obj in jdata_pre.get("objects", []):
                cat = str(obj.get("category", "")).strip().lower()
                if cat == "pin" or cat == _gas_cat:
                    continue  # gas_category already handled above
                pts_pre = np.array(obj.get("segmentation", []), dtype=np.float32)
                if pts_pre.shape[0] < 3:
                    continue
                cx_nm = float(np.mean(pts_pre[:, 0])) * nm_pre_f
                cy_nm = float(np.mean(pts_pre[:, 1])) * nm_pre_f
                area_nm2 = self.polygon_area(pts_pre) * nm_pre_f * nm_pre_f
                cat_dets[cat][int(frame_id)].append(
                    (frame_name, nm_pre_f, cx_nm, cy_nm, area_nm2)
                )

        # Build per-category ID assignments
        cat_assigned_ids = {}   # cat -> {frame_id: [id, ...]}
        cat_display_id_of = {}  # cat -> {instance_id: display_id}

        if bool(label_ids):
            for cat, det_by_frame in cat_dets.items():
                series_by_id, assigned_ids, _events = self._build_event_id_series_with_assignments(
                    det_by_frame, max_dist=float(max_dist)
                )
                cat_assigned_ids[cat] = assigned_ids
                if bool(use_display_id):
                    cat_display_id_of[cat] = self._display_id_mapping(series_by_id)
                else:
                    cat_display_id_of[cat] = None

        # ---- Per-frame annotation ----
        possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

        for frame_id, json_name in enumerate(self.json_files):
            frame_name = Path(json_name).stem

            raw_img_path = None
            for ext in possible_exts:
                p = os.path.join(raw_frame_dir, frame_name + ext)
                if os.path.exists(p):
                    raw_img_path = p
                    break
            if not raw_img_path:
                continue

            try:
                with Image.open(raw_img_path) as raw_im:
                    bg = raw_im.convert("RGBA")
                    W, H = bg.size

                    font_px = max(18, min(32, int(round(min(W, H) * 0.029))))
                    try:
                        font = ImageFont.truetype("arial.ttf", font_px)
                    except OSError:
                        font = ImageFont.load_default()
                    id_offset_x = int(max(8, round(font_px * 0.8)))
                    id_offset_y = int(max(8, round(font_px * 0.7)))
                    centroid_r = int(max(3, round(font_px * 0.22)))
                    stroke_w = int(max(1, round(font_px * 0.12)))

                    mask_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_layer)

                    json_path = os.path.join(self.json_dir, json_name)
                    with open(json_path, "r", encoding="utf-8") as f:
                        jdata = json.load(f)

                    try:
                        nm_per_px = self._nm_per_px_for_frame(frame_name)
                    except Exception:
                        nm_per_px = None

                    all_objects = jdata.get("objects", [])

                    # First pass: draw filled masks
                    for obj in all_objects:
                        cat = str(obj.get("category", "")).strip().lower()
                        if cat == "pin":
                            continue
                        pts_raw = np.array(obj.get("segmentation", []), dtype=np.float32)
                        if pts_raw.shape[0] < 3:
                            continue
                        rgb = _CATEGORY_COLOR.get(cat, _DEFAULT_COLOR)
                        fill_rgba = rgb + (mask_alpha,)
                        poly = [tuple(map(float, pt)) for pt in pts_raw]
                        mask_draw.polygon(poly, fill=fill_rgba, outline=None)

                    # Composite mask onto background
                    img_out = Image.alpha_composite(bg, mask_layer).convert("RGB")
                    draw = ImageDraw.Draw(img_out)

                    # Second pass: outlines + IDs + centroid + nanodroplet dims
                    # Track per-category object index to match pre-scan ordering
                    cat_obj_idx = defaultdict(int)

                    for obj in all_objects:
                        cat = str(obj.get("category", "")).strip().lower()
                        if cat == "pin":
                            continue
                        pts_raw = np.array(obj.get("segmentation", []), dtype=np.float32)
                        if pts_raw.shape[0] < 3:
                            cat_obj_idx[cat] += 1
                            continue

                        rgb = _CATEGORY_COLOR.get(cat, _DEFAULT_COLOR)
                        poly = [tuple(map(float, pt)) for pt in pts_raw]
                        draw.polygon(poly, outline=rgb, width=2)

                        cx_px = float(np.mean(pts_raw[:, 0]))
                        cy_px = float(np.mean(pts_raw[:, 1]))

                        # Centroid dot
                        if bool(show_centroid):
                            r = centroid_r
                            draw.ellipse((cx_px - r, cy_px - r, cx_px + r, cy_px + r),
                                         fill=rgb, outline="white", width=1)

                        # Instance ID label
                        if bool(label_ids) and cat in cat_assigned_ids:
                            obj_idx_in_cat = cat_obj_idx[cat]
                            ids_this_frame = cat_assigned_ids[cat].get(int(frame_id))
                            if ids_this_frame is not None and obj_idx_in_cat < len(ids_this_frame):
                                instance_id = int(ids_this_frame[obj_idx_in_cat])
                                display_map = cat_display_id_of.get(cat)
                                if display_map is not None:
                                    disp = int(display_map.get(instance_id, 0))
                                    id_text = str(disp) if disp > 0 else str(instance_id)
                                else:
                                    id_text = str(instance_id)
                                try:
                                    draw.text(
                                        (cx_px - id_offset_x, cy_px - id_offset_y),
                                        id_text,
                                        fill=rgb,
                                        font=font,
                                        stroke_width=stroke_w,
                                        stroke_fill="black",
                                    )
                                except Exception:
                                    pass

                        cat_obj_idx[cat] += 1

                        # Baseline + height overlay: only for nanodroplet objects
                        # AND only when the tracker's focus category is nanodroplet
                        if (cat == "nanodroplet"
                                and str(self.gas_category).lower() == "nanodroplet"
                                and nm_per_px is not None):
                            try:
                                d_px, h_px, box_info = self._compute_droplet_dims_oriented(pts_raw)
                                d_nm = float(d_px) * float(nm_per_px)
                                h_nm = float(h_px) * float(nm_per_px)

                                corners = box_info.get("corners")
                                if corners is not None:
                                    corners_arr = np.array(corners, dtype=np.float32)
                                    rect_poly = [tuple(map(float, pt)) for pt in corners_arr]
                                    draw.polygon(rect_poly, outline="cyan", width=2)
                                    text = f"D:{d_nm:.1f} {self._length_unit()}\nH:{h_nm:.1f} {self._length_unit()}"
                                    cx = float(corners_arr[:, 0].mean())
                                    cy = float(corners_arr[:, 1].mean())
                                    draw.text((cx, cy), text, fill="yellow", font=font)

                                baseline_p1 = box_info.get("baseline_p1")
                                baseline_p2 = box_info.get("baseline_p2")
                                if baseline_p1 is not None and baseline_p2 is not None:
                                    draw.line(
                                        [tuple(map(float, baseline_p1)), tuple(map(float, baseline_p2))],
                                        fill="red",
                                        width=3,
                                    )

                                apex = box_info.get("apex_point")
                                base_mid = box_info.get("base_mid_point")
                                if apex is not None and base_mid is not None:
                                    draw.line(
                                        [tuple(map(float, apex)), tuple(map(float, base_mid))],
                                        fill="magenta",
                                        width=2,
                                    )
                            except Exception:
                                pass

                    out_path = os.path.join(output_dir, frame_name + ".png")
                    img_out.save(out_path)

            except Exception as e:
                print(f"Error annotating all-categories raw frame {frame_name}: {e}")

        print(f"All-categories raw-frame annotation complete: {output_dir}")

    def export_tracked_area_results(self, tracks, out_csv=None):
        """Export tracked area series.

        CSV columns: track_id, frame_id, frame_name, nm_per_pixel, area_nm2, cx_nm, cy_nm
        """
        self._ensure_output_root()
        if out_csv is None:
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_tracked_area_vs_frame.csv")
        elif not os.path.isabs(out_csv):
             out_csv = os.path.join(self.output_root, out_csv)

        rows = []
        for track_id, t in enumerate(tracks):
            for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in t['points']:
                rows.append(
                    [track_id, frame_id, frame_name, f"{nm_per_px:.6f}", f"{area_nm2:.6f}", f"{cx_nm:.6f}", f"{cy_nm:.6f}"]
                )

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "frame_id", "frame_name", self._scale_column_name(), self._area_column_name(), self._coord_column_name("cx"), self._coord_column_name("cy")])
            writer.writerows(rows)

        print(f" - {out_csv}")

    def export_id_series(self, series_by_id, out_csv=None):
        """Export area series keyed by a globally-incrementing instance id.

        CSV columns: instance_id, frame_id, frame_name, nm_per_pixel, area_nm2, cx_nm, cy_nm
        """
        self._ensure_output_root()
        if out_csv is None:
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_instance_area_vs_frame.csv")
        elif not os.path.isabs(out_csv):
             out_csv = os.path.join(self.output_root, out_csv)

        rows = []
        for instance_id, points in series_by_id.items():
            for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in points:
                rows.append(
                    [
                        int(instance_id),
                        int(frame_id),
                        frame_name,
                        f"{float(nm_per_px):.6f}",
                        f"{float(area_nm2):.6f}",
                        f"{float(cx_nm):.6f}",
                        f"{float(cy_nm):.6f}",
                    ]
                )

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", self._scale_column_name(), self._area_column_name(), self._coord_column_name("cx"), self._coord_column_name("cy")])
            writer.writerows(rows)

        print(f" - {out_csv}")

    def export_speed_series(self, speed_series_by_id, out_csv=None):
        """Export per-instance speed series (from centroid displacement).

        Speed is computed between consecutive detections of the same instance:
            speed = distance_nm / (delta_frame * frame_interval_s)

        CSV columns: instance_id, frame_id, frame_name, speed_nm_per_s
        """
        self._ensure_output_root()
        if out_csv is None:
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_instance_speed_vs_frame.csv")
        elif not os.path.isabs(out_csv):
             out_csv = os.path.join(self.output_root, out_csv)

        rows = []
        for instance_id, points in speed_series_by_id.items():
            for frame_id, frame_name, speed_nm_per_s in points:
                rows.append([int(instance_id), int(frame_id), frame_name, f"{float(speed_nm_per_s):.6f}"])

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", f"speed_{self._length_unit()}_per_s"])
            writer.writerows(rows)

        print(f" - {out_csv}")

    @staticmethod
    def _compute_speed_series_from_points(points, frame_interval_s=1.0):
        """Compute speed series from a list of points.

        points: [(frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2), ...]
        returns: [(frame_id, frame_name, speed_nm_per_s), ...] aligned to the *current* frame.
        """
        if not points:
            return []

        dt = float(frame_interval_s)
        if dt <= 0:
            raise ValueError(f"frame_interval_s must be > 0, got {frame_interval_s}")

        # sort by frame
        pts = sorted(points, key=lambda p: int(p[0]))
        out = []
        prev = pts[0]
        for cur in pts[1:]:
            f0, _name0, _nm0, x0, y0, _a0 = prev
            f1, name1, _nm1, x1, y1, _a1 = cur
            df = int(f1) - int(f0)
            if df <= 0:
                prev = cur
                continue
            dist = float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))
            out.append((int(f1), str(name1), dist / (float(df) * dt)))
            prev = cur
        return out

    @staticmethod
    def _bin_speed_series(speed_points, bin_size_frames=10):
        """Bin speed series into non-overlapping frame windows and take the mean.

        speed_points: [(frame_id, frame_name, speed_nm_per_s), ...]
        returns: [(frame_id, frame_name, mean_speed_nm_per_s), ...]
                 where frame_id/frame_name correspond to the last point in that bin.
        """
        if not speed_points:
            return []
        b = int(bin_size_frames)
        if b <= 0:
            raise ValueError(f"bin_size_frames must be > 0, got {bin_size_frames}")

        from collections import defaultdict

        buckets = defaultdict(list)  # bin_index -> list of (frame_id, frame_name, speed)
        for frame_id, frame_name, speed in speed_points:
            idx = int(frame_id) // b
            buckets[idx].append((int(frame_id), str(frame_name), float(speed)))

        out = []
        for idx in sorted(buckets.keys()):
            items = sorted(buckets[idx], key=lambda t: t[0])
            if not items:
                continue
            frame_last, name_last, _ = items[-1]
            mean_speed = float(np.mean([s for _f, _n, s in items]))
            out.append((int(frame_last), str(name_last), mean_speed))

        return out

    @staticmethod
    def _display_id_mapping(series_by_id):
        """Map internal instance_id -> display_id (1..K) by first appearance."""
        instance_ids = sorted([int(k) for k in series_by_id.keys()])
        first_frame_by_id = {}
        for iid in instance_ids:
            pts = series_by_id.get(iid) or []
            if len(pts) == 0:
                continue
            first_frame_by_id[iid] = int(min(p[0] for p in pts))

        ordered_ids = sorted(first_frame_by_id.keys(), key=lambda i: (first_frame_by_id[i], i))
        return {iid: idx + 1 for idx, iid in enumerate(ordered_ids)}

    def _build_event_id_series(self, detections_by_frame, max_dist=50.0, return_assignments=False):
        """Assign globally-incrementing ids with continuity-first linking.

        Rules:
        - First frame detections get ids 1..N
        - Consecutive frames are linked by nearest-neighbor (one-to-one) within max_dist
        - Matched detections keep previous ids, even if object counts change
        - Unmatched current detections get NEW ids

                detections_by_frame: dict[int, list[tuple[frame_name,nm_per_px,cx_nm,cy_nm,area_nm2]]]
                returns:
                    - if return_assignments=False: (series_by_id, events)
                    - if return_assignments=True: (series_by_id, assigned_ids_by_frame, events)
        """
        from collections import defaultdict

        frames_sorted = sorted(detections_by_frame.keys())
        if not frames_sorted:
                        if bool(return_assignments):
                                return {}, {}, []
                        return {}, []

        next_id = 1
        assigned_ids_by_frame = {}
        events = []

        # init: first frame
        f0 = frames_sorted[0]
        det0 = detections_by_frame[f0]
        ids0 = []
        for _ in det0:
            ids0.append(next_id)
            next_id += 1
        assigned_ids_by_frame[f0] = ids0

        prev_dets = det0
        prev_ids = ids0

        for frame in frames_sorted[1:]:
            curr_dets = detections_by_frame[frame]
            n_prev = len(prev_dets)
            n_curr = len(curr_dets)
            curr_ids = [None] * n_curr

            if n_prev == 0:
                for j in range(n_curr):
                    curr_ids[j] = next_id
                    events.append({"frame": frame, "type": "birth", "dst_id": int(next_id)})
                    next_id += 1
                assigned_ids_by_frame[frame] = curr_ids
                prev_frame, prev_dets, prev_ids = frame, curr_dets, curr_ids
                continue

            if n_curr == 0:
                assigned_ids_by_frame[frame] = []
                prev_dets, prev_ids = curr_dets, []
                continue

            # Do one-to-one assignment by minimal distance for all count combinations.
            prev_xy = np.array([[d[2], d[3]] for d in prev_dets], dtype=np.float64)
            curr_xy = np.array([[d[2], d[3]] for d in curr_dets], dtype=np.float64)
            dists = np.linalg.norm(prev_xy[:, None, :] - curr_xy[None, :, :], axis=2)

            pairs = []  # (dist, i_prev, j_curr)
            for i in range(n_prev):
                for j in range(n_curr):
                    dist = float(dists[i, j])
                    if dist <= float(max_dist):
                        pairs.append((dist, i, j))
            pairs.sort(key=lambda x: x[0])

            used_prev = set()
            used_curr = set()
            for _dist, i, j in pairs:
                if i in used_prev or j in used_curr:
                    continue
                curr_ids[j] = int(prev_ids[i])
                used_prev.add(i)
                used_curr.add(j)

            # any unmatched current object becomes a new id
            for j in range(n_curr):
                if curr_ids[j] is None:
                    curr_ids[j] = int(next_id)
                    events.append({"frame": frame, "type": "birth", "dst_id": int(next_id)})
                    next_id += 1

            assigned_ids_by_frame[frame] = curr_ids
            prev_dets, prev_ids = curr_dets, curr_ids

        # build series
        series_by_id = defaultdict(list)
        for frame in frames_sorted:
            dets = detections_by_frame[frame]
            ids = assigned_ids_by_frame.get(frame, [])
            for det, instance_id in zip(dets, ids):
                frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 = det
                series_by_id[int(instance_id)].append((int(frame), frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2)))

        if bool(return_assignments):
            return dict(series_by_id), assigned_ids_by_frame, events
        return dict(series_by_id), events

    def _build_event_id_series_with_assignments(self, detections_by_frame, max_dist=50.0):
        """Compatibility helper: return series + per-frame assignment list + events."""
        return self._build_event_id_series(detections_by_frame, max_dist=max_dist, return_assignments=True)

    def _build_greedy_tracks(self, detections_by_frame, max_dist=50.0):
        """Greedy link detections in consecutive frames into tracks.

        detections_by_frame: dict[int, list[tuple[frame_name,nm_per_px,cx_nm,cy_nm,area_nm2]]]
        max_dist: distance threshold in nm
        returns: list of tracks, each track: {'last_frame': int, 'points': [(frame_id,frame_name,nm_per_px,cx_nm,cy_nm,area_nm2), ...]}
        """
        tracks = []
        for frame in sorted(detections_by_frame.keys()):
            dets = detections_by_frame[frame]
            assigned = [False] * len(dets)

            # extend tracks from previous frame
            for t in tracks:
                if t['last_frame'] != frame - 1:
                    continue

                last_x, last_y = t['points'][-1][3], t['points'][-1][4]
                best_idx = None
                best_dist = float('inf')
                for i, (frame_name, nm_per_px, cx_nm, cy_nm, area_nm2) in enumerate(dets):
                    if assigned[i]:
                        continue
                    d = np.hypot(cx_nm - last_x, cy_nm - last_y)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_idx is not None and best_dist <= max_dist:
                    frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 = dets[best_idx]
                    t['points'].append((frame, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2))
                    t['last_frame'] = frame
                    assigned[best_idx] = True

            # create new tracks for unassigned detections
            for i, (frame_name, nm_per_px, cx_nm, cy_nm, area_nm2) in enumerate(dets):
                if not assigned[i]:
                    tracks.append({'last_frame': frame, 'points': [(frame, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2)]})

        return tracks

    def plot_area_trajectories(self, max_dist=50.0, min_track_length=1, outname=None, id_mode="event", debug_stats=False):
        """Plot each droplet's area-vs-frame curve in one figure.

        Tracks are built by greedy centroid linking.
        NOTE: max_dist is in nm because centroids are stored in nm.
        """
        self._ensure_output_root()
        if len(self.object_records) == 0:
            print("No object records to plot area trajectories.")
            return

        from collections import defaultdict

        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
            by_frame[int(frame_id)].append((frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2)))

        if bool(debug_stats):
            n_frames = len(by_frame)
            n_dets = sum(len(v) for v in by_frame.values())
            print(
                f"[debug] {self.gas_category} area: frames_with_detections={n_frames}, total_detections={n_dets}, "
                f"id_mode={id_mode}, {self._max_dist_label()}={float(max_dist)}, min_track_length={int(min_track_length)}"
            )

        if str(id_mode).lower() == "greedy":
            tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} area: greedy tracks before length filter={len(tracks)}")
            tracks = [t for t in tracks if len(t['points']) >= int(min_track_length)]
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} area: greedy tracks after length filter={len(tracks)}")
            series_by_id = {int(track_id): [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in t["points"]] for track_id, t in enumerate(tracks)}
        else:
            series_by_id, _events = self._build_event_id_series(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} area: event ids before length filter={len(series_by_id)}")
            series_by_id = {k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)}
            if bool(debug_stats):
                kept = len(series_by_id)
                max_iid = max(series_by_id.keys()) if kept > 0 else None
                print(f"[debug] {self.gas_category} area: event ids after length filter={kept}, max_instance_id={max_iid}")

        if outname is None:
            outname = os.path.join(self.output_root, f"{self.gas_category}_area_trajectories.png")
        elif not os.path.isabs(outname):
            outname = os.path.join(self.output_root, outname)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Frame id")
        ax.set_ylabel(f"Area ({self._area_unit()})")
        ax.grid(True, alpha=0.25)

        # color cycle (good for dozens of tracks; for hundreds, they'll repeat)
        cmap = plt.cm.tab20

        display_id_of = self._display_id_mapping(series_by_id)
        instance_ids = sorted(series_by_id.keys())

        if bool(debug_stats):
            max_disp = max(display_id_of.values()) if len(display_id_of) > 0 else 0
            print(f"[debug] {self.gas_category} area: plotted_ids={len(instance_ids)}, display_id_max={max_disp}")

        line_handles = []
        line_labels = []

        skipped_empty = []

        for instance_id in instance_ids:
            pts = series_by_id[instance_id]
            frames = np.array([p[0] for p in pts], dtype=np.int32)
            areas = np.array([p[5] for p in pts], dtype=np.float32)
            if frames.size == 0:
                if bool(debug_stats):
                    skipped_empty.append(int(instance_id))
                continue
            order = np.argsort(frames)
            frames = frames[order]
            areas = areas[order]

            disp_id = display_id_of.get(int(instance_id), 0)
            color = cmap(int(disp_id) % 20)
            (line,) = ax.plot(frames, areas, color=color, linewidth=1.2, alpha=0.85)
            line_handles.append(line)
            line_id_label = str(int(disp_id) if disp_id > 0 else int(instance_id))
            line_labels.append(line_id_label)

            # Mark ID at the start of each curve
            try:
                x0 = float(frames[0])
                y0 = float(areas[0])
                ax.annotate(
                    line_id_label,
                    xy=(x0, y0),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=12,
                    color=color,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
                )
            except Exception:
                pass

        if bool(debug_stats):
            if len(skipped_empty) > 0:
                print(f"[debug] {self.gas_category} area: skipped_empty_ids={len(skipped_empty)} head={skipped_empty[:20]}")
            print(f"[debug] {self.gas_category} area: drawn_lines={len(line_handles)} legend_items={len(line_labels)}")

        # 图例：自适应布局，避免挡线 & 避免图被挤得很“扁”
        leg = None
        if len(line_handles) > 0:
            n_items = len(line_handles)

            # Prefer right-side legend for moderate counts; switch to multi-column / bottom for very long legends.
            if n_items <= 20:
                ncol = 1
                fig.set_size_inches(12, 6, forward=True)
                fig.subplots_adjust(right=0.80)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=14,
                    ncol=ncol,
                )
            elif n_items <= 60:
                ncol = 2
                fig.set_size_inches(14, 6, forward=True)
                fig.subplots_adjust(right=0.78)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=13,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )
            else:
                # Too many: put legend below with more columns to avoid clipping.
                # Aim for <= ~12 rows in legend.
                rows_target = 12
                ncol = int(np.ceil(float(n_items) / float(rows_target)))
                ncol = max(4, min(10, ncol))
                fig.set_size_inches(16, 11.0, forward=True)
                fig.subplots_adjust(bottom=0.36)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.22),
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=12,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )

        ax.set_title(
            f"{self.gas_category}: area vs frame (per droplet track) | tracks={len(instance_ids)}",
            loc="center",
        )
        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight", bbox_extra_artists=((leg,) if leg is not None else None))
        print(f"Saved area trajectories plot: {outname}")

        # also export tracked series for downstream analysis
        if str(id_mode).lower() == "greedy":
            self.export_tracked_area_results(tracks)
        else:
            self.export_id_series(series_by_id)

    def plot_velocity_trajectories(
        self,
        max_dist=50.0,
        min_track_length=1,
        outname=None,
        id_mode="event",
        frame_interval_s=1.0,
        bin_size_frames=10,
        debug_stats=False,
    ):
        """Plot each individual's speed-vs-frame curve.

        Speed is computed from centroid displacement between consecutive detections.
        NOTE: speed unit is nm/s; set frame_interval_s (seconds per frame) to match your acquisition.
        """
        self._ensure_output_root()
        if len(self.object_records) == 0:
            print("No object records to plot velocity trajectories.")
            return

        from collections import defaultdict

        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
            by_frame[int(frame_id)].append((frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2)))

        if bool(debug_stats):
            n_frames = len(by_frame)
            n_dets = sum(len(v) for v in by_frame.values())
            print(
                f"[debug] {self.gas_category} speed: frames_with_detections={n_frames}, total_detections={n_dets}, "
                f"id_mode={id_mode}, {self._max_dist_label()}={float(max_dist)}, min_track_length={int(min_track_length)}, "
                f"frame_interval_s={float(frame_interval_s)}, bin_size_frames={int(bin_size_frames)}"
            )

        if str(id_mode).lower() == "greedy":
            tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} speed: greedy tracks before length filter={len(tracks)}")
            tracks = [t for t in tracks if len(t["points"]) >= int(min_track_length)]
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} speed: greedy tracks after length filter={len(tracks)}")
            series_by_id = {
                int(track_id): [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in t["points"]]
                for track_id, t in enumerate(tracks)
            }
        else:
            series_by_id, _events = self._build_event_id_series(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} speed: event ids before length filter={len(series_by_id)}")
            series_by_id = {k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)}
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} speed: event ids after length filter={len(series_by_id)}")

        # compute speed series for each id (raw, per-frame)
        speed_series_by_id = {}
        empty_speed_ids = 0
        for instance_id, pts in series_by_id.items():
            sp = self._compute_speed_series_from_points(pts, frame_interval_s=frame_interval_s)
            if len(sp) > 0:
                speed_series_by_id[int(instance_id)] = sp
            else:
                empty_speed_ids += 1

        if bool(debug_stats):
            print(
                f"[debug] {self.gas_category} speed: ids_with_speed={len(speed_series_by_id)}, "
                f"ids_dropped_empty_speed={int(empty_speed_ids)} (typically tracks with <2 detections)"
            )

        # bin-average: every N frames a mean value
        b = int(bin_size_frames)
        if b <= 0:
            raise ValueError(f"bin_size_frames must be > 0, got {bin_size_frames}")

        if b == 1:
            binned_speed_by_id = dict(speed_series_by_id)
        else:
            binned_speed_by_id = {}
            for instance_id, sp in speed_series_by_id.items():
                bp = self._bin_speed_series(sp, bin_size_frames=b)
                if len(bp) > 0:
                    binned_speed_by_id[int(instance_id)] = bp

        if bool(debug_stats):
            max_disp = 0
            if len(binned_speed_by_id) > 0:
                display_id_of_dbg = self._display_id_mapping(binned_speed_by_id)
                max_disp = max(display_id_of_dbg.values()) if len(display_id_of_dbg) > 0 else 0
            print(f"[debug] {self.gas_category} speed: plotted_ids={len(binned_speed_by_id)}, display_id_max={max_disp}")

        if outname is None:
            if b == 1:
                outname = os.path.join(self.output_root, f"{self.gas_category}_velocity_trajectories.png")
            else:
                outname = os.path.join(self.output_root, f"{self.gas_category}_velocity_mean_{b}frames.png")
        elif not os.path.isabs(outname):
            outname = os.path.join(self.output_root, outname)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Frame id")
        ax.set_ylabel(f"Speed ({self._speed_unit()})")
        ax.grid(True, alpha=0.25)

        cmap = plt.cm.tab20
        display_id_of = self._display_id_mapping(binned_speed_by_id)
        instance_ids = sorted(binned_speed_by_id.keys())

        line_handles = []
        line_labels = []

        for instance_id in instance_ids:
            pts = binned_speed_by_id[instance_id]
            frames = np.array([p[0] for p in pts], dtype=np.int32)
            speeds = np.array([p[2] for p in pts], dtype=np.float32)
            if frames.size == 0:
                continue
            order = np.argsort(frames)
            frames = frames[order]
            speeds = speeds[order]

            disp_id = display_id_of.get(int(instance_id), 0)
            color = cmap(int(disp_id) % 20)
            (line,) = ax.plot(frames, speeds, color=color, linewidth=1.2, alpha=0.85)
            line_handles.append(line)
            line_id_label = str(int(disp_id) if disp_id > 0 else int(instance_id))
            line_labels.append(line_id_label)

            # Mark ID at the start of each curve
            try:
                x0 = float(frames[0])
                y0 = float(speeds[0])
                ax.annotate(
                    line_id_label,
                    xy=(x0, y0),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=12,
                    color=color,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
                )
            except Exception:
                pass

        # legend layout (same idea as area plot)
        leg = None
        if len(line_handles) > 0:
            n_items = len(line_handles)
            if n_items <= 20:
                fig.set_size_inches(12, 6, forward=True)
                fig.subplots_adjust(right=0.80)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=14,
                    ncol=1,
                )
            elif n_items <= 60:
                fig.set_size_inches(14, 6, forward=True)
                fig.subplots_adjust(right=0.78)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=13,
                    ncol=2,
                    columnspacing=0.8,
                    handlelength=1.2,
                )
            else:
                rows_target = 12
                ncol = int(np.ceil(float(n_items) / float(rows_target)))
                ncol = max(4, min(10, ncol))
                fig.set_size_inches(16, 11.0, forward=True)
                fig.subplots_adjust(bottom=0.36)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.22),
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=12,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )

        if b == 1:
            ax.set_title(f"{self.gas_category}: velocity vs frame (per track) | tracks={len(instance_ids)}", loc="center")
        else:
            ax.set_title(f"{self.gas_category}: mean velocity per {b} frames | tracks={len(instance_ids)}", loc="center")
        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight", bbox_extra_artists=((leg,) if leg is not None else None))
        print(f"Saved velocity trajectories plot: {outname}")

        # export for downstream analysis
        if b == 1:
            self.export_speed_series(speed_series_by_id)
        else:
            self.export_speed_series(
                binned_speed_by_id,
                out_csv=f"{self.gas_category}_instance_speed_mean_{b}frames.csv",
            )

    # Alias for naming preference
    def plot_speed_trajectories(self, *args, **kwargs):
        return self.plot_velocity_trajectories(*args, **kwargs)

    def plot_area_delta_vs_frame(
        self,
        outname=None,
        out_csv=None,
        per_frame=True,
        reducer="sum",
    ):
        """Plot per-frame change in area (Δarea) as a single curve.

        This is computed from `self.area_records` by first aggregating all detections within
        the same frame (default: sum), then taking a first-order difference between
        consecutive frames.

        Args:
            outname: Output PNG name.
            out_csv: Optional CSV output for the delta series.
            per_frame: If True, normalize by delta_frame (handles skipped frames).
            reducer: How to aggregate multiple objects in the same frame: 'sum' or 'mean'.
        """
        self._ensure_output_root()
        if len(self.area_records) == 0:
            print("No area records to plot area delta.")
            return

        from collections import defaultdict

        reducer_key = str(reducer).strip().lower()
        if reducer_key not in {"sum", "mean"}:
            raise ValueError(f"reducer must be 'sum' or 'mean', got {reducer}")

        areas_by_frame = defaultdict(list)  # frame_id -> list[area_nm2]
        name_by_frame = {}
        for frame_id, frame_name, _nm_per_px, area_nm2 in self.area_records:
            fid = int(frame_id)
            areas_by_frame[fid].append(float(area_nm2))
            if fid not in name_by_frame:
                name_by_frame[fid] = str(frame_name)

        frame_ids = sorted(areas_by_frame.keys())
        if len(frame_ids) < 2:
            print("Not enough frames to compute area delta (need >= 2).")
            return

        area_series = []  # (frame_id, frame_name, area_agg_nm2)
        for fid in frame_ids:
            vals = areas_by_frame[fid]
            if len(vals) == 0:
                continue
            if reducer_key == "mean":
                a = float(np.mean(vals))
            else:
                a = float(np.sum(vals))
            area_series.append((int(fid), name_by_frame.get(int(fid), str(fid)), a))

        # ensure sorted
        area_series.sort(key=lambda t: int(t[0]))

        # delta aligned to current frame
        delta_points = []  # (frame_id, frame_name, delta_area_nm2_per_frame)
        prev_f, _prev_name, prev_a = area_series[0]
        for cur_f, cur_name, cur_a in area_series[1:]:
            df = int(cur_f) - int(prev_f)
            if df <= 0:
                prev_f, prev_a = cur_f, cur_a
                continue
            da = float(cur_a) - float(prev_a)
            if bool(per_frame):
                da = da / float(df)
            delta_points.append((int(cur_f), str(cur_name), float(da)))
            prev_f, prev_a = cur_f, cur_a

        if len(delta_points) == 0:
            print("Area delta series is empty after processing.")
            return

        if outname is None:
            outname = os.path.join(self.output_root, f"{self.gas_category}_area_delta_vs_frame.png")
        elif not os.path.isabs(outname):
            outname = os.path.join(self.output_root, outname)

        if out_csv is None:
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_area_delta_vs_frame.csv")
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        frames = np.array([p[0] for p in delta_points], dtype=np.int32)
        deltas = np.array([p[2] for p in delta_points], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, deltas, color="#1f77b4", linewidth=1.6)
        ax.set_ylabel(f"Speed ({self._speed_unit()})")
        ax.set_xlabel("Frame id")
        ylab = f"ΔArea ({self._area_unit()}/frame)" if bool(per_frame) else f"ΔArea ({self._area_unit()})"
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)

        agg_label = "sum" if reducer_key == "sum" else "mean"
        ax.set_title(f"{self.gas_category}: per-frame area change (Δarea), frame-agg={agg_label}", loc="center")

        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved area delta plot: {outname}")

        # export delta CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "frame_name", f"{self._area_column_name('delta_area')}_per_frame" if bool(per_frame) else self._area_column_name("delta_area")])
            for fid, fname, da in delta_points:
                writer.writerow([int(fid), str(fname), f"{float(da):.6f}"])
        print(f" - {out_csv}")

    def plot_frame_instance_count_and_total_area(self, outname=None, out_csv=None):
        """Plot per-frame instance count and total area as two separate figures.

        Uses `self.area_records` where each row is one detected instance in a frame.
        - instance_count(frame): number of instances in this frame
        - total_area_nm2(frame): sum of all instance areas in this frame
        """
        self._ensure_output_root()
        if len(self.area_records) == 0:
            print("No area records to plot frame totals.")
            return

        from collections import defaultdict

        count_by_frame = defaultdict(int)
        area_sum_by_frame = defaultdict(float)
        name_by_frame = {}

        for frame_id, frame_name, _nm_per_px, area_nm2 in self.area_records:
            fid = int(frame_id)
            count_by_frame[fid] += 1
            area_sum_by_frame[fid] += float(area_nm2)
            if fid not in name_by_frame:
                name_by_frame[fid] = str(frame_name)

        frame_ids = sorted(count_by_frame.keys())
        if len(frame_ids) == 0:
            print("No valid frame statistics to plot.")
            return

        if outname is None:
            count_plot_path = os.path.join(self.output_root, f"{self.gas_category}_frame_instance_count.png")
            area_plot_path = os.path.join(self.output_root, f"{self.gas_category}_frame_total_area.png")
        else:
            # If outname is provided, treat it as a shared prefix for two plot files.
            if not os.path.isabs(outname):
                outname = os.path.join(self.output_root, outname)
            base, ext = os.path.splitext(outname)
            if ext == "":
                ext = ".png"
            count_plot_path = f"{base}_instance_count{ext}"
            area_plot_path = f"{base}_total_area{ext}"

        if out_csv is None:
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_frame_count_area.csv")
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        frames = np.array(frame_ids, dtype=np.int32)
        counts = np.array([count_by_frame[fid] for fid in frame_ids], dtype=np.int32)
        areas = np.array([area_sum_by_frame[fid] for fid in frame_ids], dtype=np.float64)

        # Plot 1: instance count only
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, counts, color="#1f77b4", linewidth=1.8)
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Instance count")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{self.gas_category}: per-frame instance count", loc="center")
        plt.tight_layout()
        plt.savefig(count_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved frame instance-count plot: {count_plot_path}")
        plt.close(fig)

        # Plot 2: total area only
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, areas, color="#d62728", linewidth=1.8)
        ax.set_xlabel("Frame id")
        ax.set_ylabel(f"Total area ({self._area_unit()})")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{self.gas_category}: per-frame total area", loc="center")
        plt.tight_layout()
        plt.savefig(area_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved frame total-area plot: {area_plot_path}")
        plt.close(fig)

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "frame_name", "instance_count", self._area_column_name("total_area")])
            for fid in frame_ids:
                writer.writerow([
                    int(fid),
                    name_by_frame.get(int(fid), str(fid)),
                    int(count_by_frame[fid]),
                    f"{float(area_sum_by_frame[fid]):.6f}",
                ])
        print(f" - {out_csv}")
    # -----------------------------
    # 可视化（抽帧）
    # -----------------------------
    def plot_evolution(self, step=200, max_dist=50.0, min_track_length=1):
        self._ensure_output_root()
        if self.W is None or self.H is None:
            print("[skip] plot_evolution: no image size is available from raw_frame_dir/image_path.")
            return

        allowed_instance_ids = None
        if int(min_track_length) > 1 and len(self.object_records) > 0:
            aligned_instance_ids = self._build_export_instance_ids(
                max_dist=max_dist,
                id_mode="event",
                use_display_id=False,
            )
            counts_by_id = Counter(int(instance_id) for instance_id in aligned_instance_ids)
            allowed_instance_ids = {
                int(instance_id)
                for instance_id, count in counts_by_id.items()
                if int(count) >= int(min_track_length)
            }
        elif len(self.object_records) > 0:
            aligned_instance_ids = self._build_export_instance_ids(
                max_dist=max_dist,
                id_mode="event",
                use_display_id=False,
            )
        else:
            aligned_instance_ids = []

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_aspect("equal", adjustable="box")

        cmap = plt.cm.plasma
        norm = Normalize(vmin=0, vmax=len(self.json_files) - 1)

        for row_idx, row in enumerate(self.contour_records):
            frame_id = row[0]
            if frame_id % step != 0:
                continue
            if allowed_instance_ids is not None:
                if row_idx >= len(aligned_instance_ids):
                    continue
                if int(aligned_instance_ids[row_idx]) not in allowed_instance_ids:
                    continue

            pts = []
            # row format: [frame_id, frame_name, "(x_nm,y_nm)", ...]
            for item in row[2:]:
                x, y = map(
                    float,
                    item.strip("()").split(",")
                )
                pts.append([x, y])

            pts = np.array(pts)
            pts = self._coords_nm_to_plot_px(pts, frame_name=row[1])
            pts = np.vstack([pts, pts[0]])

            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=cmap(norm(frame_id)),
                linewidth=1.5,
                alpha=0.85
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Frame id colorbar: same height as the axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Frame id")

        ax.set_title(f"{self.gas_category} domain evolution (pin-referenced)", loc="center")
        plt.tight_layout()
        # add a visible border around the axes
        from matplotlib.patches import Rectangle
        border_width = 3
        border_color = "black"
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                 fill=False, edgecolor=border_color,
                 linewidth=border_width, zorder=10, clip_on=False)
        ax.add_patch(rect)
        outname = os.path.join(self.output_root, f"{self.gas_category}_evolution.png")
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved evolution plot: {outname}")

    def plot_centroid_trajectories(self, max_dist=50.0, min_track_length=1):
        """
        Build simple greedy tracks by linking centroids in consecutive frames
        when their distance is <= max_dist. Save plot to PNG.
        NOTE: max_dist is in the current exported distance unit.
        """
        self._ensure_output_root()
        if self.W is None or self.H is None:
            print("[skip] plot_centroid_trajectories: no image size is available from raw_frame_dir/image_path.")
            return

        if len(self.centroid_records) == 0:
            print("No centroid records to plot.")
            return

        by_frame = self._object_detections_by_frame()
        series_by_id, _events = self._build_event_id_series(by_frame, max_dist=max_dist)
        series_by_id = {
            int(instance_id): pts
            for instance_id, pts in series_by_id.items()
            if len(pts) >= int(min_track_length)
        }

        # plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_aspect("equal", adjustable="box")

        # color by frame (time axis) — use same colormap/norm as evolution
        cmap = plt.cm.plasma
        norm = Normalize(vmin=0, vmax=len(self.json_files) - 1)

        for instance_id, series in series_by_id.items():
            frames = np.array([p[0] for p in series])
            pts = np.array([[p[3], p[4]] for p in series], dtype=np.float64)
            nm_scales = [p[2] for p in series]
            frame_names = [p[1] for p in series]
            pts = np.array([
                self._coords_nm_to_plot_px(pt, frame_name=frame_name, nm_per_px=nm_per_px)
                for pt, frame_name, nm_per_px in zip(pts, frame_names, nm_scales)
            ], dtype=np.float64)
            if pts.shape[0] == 0:
                continue

            # draw colored segments between consecutive points according to the earlier frame
            for i in range(len(pts) - 1):
                col = cmap(norm(frames[i]))
                ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], '-', color=col, linewidth=1, alpha=0.95)

            # scatter points colored by their frame
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=frames, cmap=cmap, norm=norm, s=1)

        # add colorbar (time axis)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Frame id colorbar: same height as the axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Frame id")

        ax.set_title(f"{self.gas_category} centroid trajectories (time-colored)", loc="center")
        plt.tight_layout()
        outname = os.path.join(self.output_root, f"{self.gas_category}_centroid_trajectories.png")
        # add a visible border around the axes
        from matplotlib.patches import Rectangle
        border_width = 3
        border_color = "black"
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                 fill=False, edgecolor=border_color,
                 linewidth=border_width, zorder=10, clip_on=False)
        ax.add_patch(rect)
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved centroid trajectories plot: {outname}")


# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
    gas_category = "nanocluster"
    output_root = os.path.join("outputs/zwl", gas_category)
    raw_frame_dir = "data_cus/zwl_resize512_878/group_0/frame"
    manual_scale_pixel_length =10   # e.g. 120.0 means the 20 nm scale bar spans 120 px
    manual_nm_per_px = None            # e.g. 0.166667; overrides manual_scale_pixel_length if set alone
    min_track_length_plot = 3          # filter short-lived detections in trajectory plots

    tracker = GasTracker(
        json_dir="outputs/zwl_resize512_878-sam3/group_0/mark",
        raw_frame_dir=raw_frame_dir,
        #scale_csv=r"D:\code\nanojccode\data\nanoframes\scalebar_mauel.csv",
        output_root=output_root,
        scale_value_nm=20.0,
        manual_scale_pixel_length=manual_scale_pixel_length,
        manual_nm_per_px=manual_nm_per_px,
        strict_scale_match=False,
        gas_category=gas_category,
        #pin_category="pin"
    )
    tracker.process_all_frames()
    tracker.export_results()
    tracker.annotate_images(label_ids=True)
    
    tracker.annotate_images_on_rawframe(
        raw_frame_dir=raw_frame_dir,   # 原始帧图像所在目录
        label_ids=True,
        mask_alpha=120,
    )
    tracker.annotate_allcategories_on_rawframe(
        raw_frame_dir=raw_frame_dir,   # 原始帧图像所在目录
        mask_alpha=120,
        show_centroid=False,
        label_ids=False
    )
    tracker.plot_evolution(step=2, max_dist=50, min_track_length=min_track_length_plot)
    tracker.plot_centroid_trajectories(max_dist=50, min_track_length=min_track_length_plot)
    tracker.plot_area_trajectories(max_dist=50, min_track_length=min_track_length_plot, debug_stats=True)
    tracker.plot_frame_instance_count_and_total_area()
    tracker.plot_area_delta_vs_frame(per_frame=True, reducer="sum")
    # 30 fps => 1/30 s per frame; speed unit: nm/s
    tracker.plot_velocity_trajectories(
        max_dist=50,
        min_track_length=min_track_length_plot,
        frame_interval_s=1/30,
        bin_size_frames=1,
        debug_stats=True,
    )
