from flask import Flask, render_template, request, send_file, send_from_directory, redirect, url_for
import os
import uuid
import io
import zipfile
import random
from math import sqrt
from PIL import Image
from svg_separator import separate_color_layers

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_rgba(file_storage) -> Image.Image:
    img = Image.open(file_storage).convert("RGBA")
    return img


def get_pixels_rgba(img: Image.Image):
    w, h = img.size
    return list(img.getdata()), w, h


def rgb_to_lab_approx(r, g, b):
    r_ = r / 255.0
    g_ = g / 255.0
    b_ = b / 255.0
    x = 0.4124564 * r_ + 0.3575761 * g_ + 0.1804375 * b_
    y = 0.2126729 * r_ + 0.7151522 * g_ + 0.0721750 * b_
    z = 0.0193339 * r_ + 0.1191920 * g_ + 0.9503041 * b_
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16 / 116)
    fx = f(x)
    fy = f(y)
    fz = f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return (L, a, b)


def kmeans_points(points, k, max_iter=15):
    if not points:
        return [], []
    k = max(1, min(k, len(points)))
    centers = random.sample(points, k)
    labels = [0] * len(points)
    for _ in range(max_iter):
        changed = False
        for i, p in enumerate(points):
            best = 0
            bestd = 1e18
            for ci, c in enumerate(centers):
                d = (p[0]-c[0])**2 + (p[1]-c[1])**2 + (p[2]-c[2])**2
                if d < bestd:
                    bestd = d
                    best = ci
            if labels[i] != best:
                labels[i] = best
                changed = True
        if not changed:
            break
        sums = [(0.0, 0.0, 0.0, 0) for _ in centers]
        for p, li in zip(points, labels):
            sL, sA, sB, n = sums[li]
            sums[li] = (sL + p[0], sA + p[1], sB + p[2], n + 1)
        new_centers = []
        for sL, sA, sB, n in sums:
            if n == 0:
                new_centers.append(random.choice(points))
            else:
                new_centers.append((sL / n, sA / n, sB / n))
        centers = new_centers
    return centers, labels


def compute_sse(points, centers, labels):
    sse = 0.0
    for p, li in zip(points, labels):
        c = centers[li]
        sse += (p[0]-c[0])**2 + (p[1]-c[1])**2 + (p[2]-c[2])**2
    return sse


def sample_lab_points_from_pixels(pixels, max_points=8000):
    pts = []
    for (r, g, b, a) in pixels:
        if a == 0:
            continue
        L, A, B = rgb_to_lab_approx(r, g, b)
        pts.append((L, A, B))
    if len(pts) > max_points:
        pts = random.sample(pts, max_points)
    return pts


def estimate_optimal_k(points, max_k=8, improvement_threshold=0.15, min_k=2):
    if not points:
        return 1
    max_k = max(1, min(max_k, len(points)))
    prev_sse = None
    best_k = 1
    for k in range(1, max_k + 1):
        centers, labels = kmeans_points(points, k)
        sse = compute_sse(points, centers, labels)
        if prev_sse is not None:
            drop = (prev_sse - sse) / prev_sse if prev_sse > 0 else 0.0
            if k >= min_k and drop < improvement_threshold:
                best_k = k - 1 if k - 1 >= 1 else 1
                break
        prev_sse = sse
        best_k = k
    return best_k


def assign_full_image_labels(img: Image.Image, centers):
    w, h = img.size
    px = img.load()
    label_img = [[-1] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            L, A, B = rgb_to_lab_approx(r, g, b)
            best = 0
            bestd = 1e18
            for ci, c in enumerate(centers):
                d = (L-c[0])**2 + (A-c[1])**2 + (B-c[2])**2
                if d < bestd:
                    bestd = d
                    best = ci
            label_img[y][x] = best
    return label_img


def build_masks(label_img, k):
    h = len(label_img)
    w = len(label_img[0]) if h else 0
    masks = []
    for i in range(k):
        m = [[0] * w for _ in range(h)]
        for y in range(h):
            row = label_img[y]
            for x in range(w):
                if row[x] == i:
                    m[y][x] = 1
        masks.append(m)
    return masks


def connected_component_filter(mask, min_area):
    h = len(mask)
    w = len(mask[0]) if h else 0
    visited = [[False]*w for _ in range(h)]
    out = [[0]*w for _ in range(h)]
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for y in range(h):
        for x in range(w):
            if mask[y][x] == 1 and not visited[y][x]:
                stack = [(x,y)]
                comp = []
                visited[y][x] = True
                while stack:
                    cx, cy = stack.pop()
                    comp.append((cx, cy))
                    for dx, dy in dirs:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx] and mask[ny][nx] == 1:
                            visited[ny][nx] = True
                            stack.append((nx, ny))
                if len(comp) >= min_area:
                    for cx, cy in comp:
                        out[cy][cx] = 1
    return out


def dilate(mask, radius):
    if radius <= 0:
        return mask
    h = len(mask)
    w = len(mask[0]) if h else 0
    out = [[0]*w for _ in range(h)]
    r = radius
    rr = r*r
    offsets = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dx*dx + dy*dy <= rr:
                offsets.append((dx, dy))
    for y in range(h):
        for x in range(w):
            v = 0
            for dx, dy in offsets:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] == 1:
                    v = 1
                    break
            out[y][x] = v
    return out


def mask_to_rgba_image(mask, w, h, color, alpha):
    img = Image.new("RGBA", (w, h))
    px = img.load()
    r, g, b = color
    for y in range(h):
        for x in range(w):
            if mask[y][x] == 1:
                px[x, y] = (r, g, b, alpha)
            else:
                px[x, y] = (0, 0, 0, 0)
    return img


def mask_from_source(mask, source_img, use_source_alpha=True):
    w, h = source_img.size
    src_px = source_img.load()
    out = Image.new("RGBA", (w, h))
    out_px = out.load()
    for y in range(h):
        for x in range(w):
            if mask[y][x] == 1:
                r, g, b, a = src_px[x, y]
                out_px[x, y] = (r, g, b, a if use_source_alpha else 255)
            else:
                out_px[x, y] = (0, 0, 0, 0)
    return out


def lab_to_rgb_approx(L, A, B):
    fy = (L + 16) / 116.0
    fx = fy + (A / 500.0)
    fz = fy - (B / 200.0)
    def invf(t):
        return t**3 if t**3 > 0.008856 else (t - 16/116) / 7.787
    x = invf(fx)
    y = invf(fy)
    z = invf(fz)
    r_ = 3.2404542*x - 1.5371385*y - 0.4985314*z
    g_ = -0.9692660*x + 1.8760108*y + 0.0415560*z
    b_ = 0.0556434*x - 0.2040259*y + 1.0572252*z
    def clamp01(v):
        return 0 if v < 0 else 1 if v > 1 else v
    r = int(clamp01(r_) * 255 + 0.5)
    g = int(clamp01(g_) * 255 + 0.5)
    b = int(clamp01(b_) * 255 + 0.5)
    return (r, g, b)


def build_underbase(masks, spread_px):
    if not masks:
        return None
    h = len(masks[0])
    w = len(masks[0][0]) if h else 0
    ub = [[0]*w for _ in range(h)]
    for m in masks:
        for y in range(h):
            row_m = m[y]
            row_u = ub[y]
            for x in range(w):
                if row_m[x] == 1:
                    row_u[x] = 1
    ub = dilate(ub, spread_px)
    return ub


@app.route("/")
def index():
    return render_template("index.html", result=None)


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))
    plate_expand_px = int(request.form.get("plate_expand_px", 0))
    min_area = int(request.form.get("min_area", 16))
    include_underbase = request.form.get("include_underbase") == "on"
    underbase_spread_px = int(request.form.get("underbase_spread_px", 2))
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    original = load_image_rgba(file)
    pixels, w, h = get_pixels_rgba(original)
    sample_points = sample_lab_points_from_pixels(pixels, max_points=8000)
    k = estimate_optimal_k(sample_points, max_k=8, improvement_threshold=0.15, min_k=2)
    centers, _ = kmeans_points(sample_points, k)
    if not centers:
        return render_template("index.html", result=None)
    label_img = assign_full_image_labels(original, centers)
    base_masks = build_masks(label_img, len(centers))
    prod_masks = []
    for m in base_masks:
        mm = m
        if min_area > 1:
            mm = connected_component_filter(mm, min_area)
        if plate_expand_px > 0:
            mm = dilate(mm, plate_expand_px)
        prod_masks.append(mm)
    areas = []
    for m in base_masks:
        a = 0
        for y in range(h):
            for x in range(w):
                if m[y][x] == 1:
                    a += 1
        areas.append(a)
    valid = [i for i, a in enumerate(areas) if a > 0]
    base_masks = [base_masks[i] for i in valid]
    prod_masks = [prod_masks[i] for i in valid]
    centers = [centers[i] for i in valid]
    if not base_masks:
        return render_template("index.html", result=None)
    order = list(range(len(base_masks)))
    order.sort(key=lambda i: sum(1 for y in range(h) for x in range(w) if base_masks[i][y][x] == 1), reverse=True)
    base_masks = [base_masks[i] for i in order]
    prod_masks = [prod_masks[i] for i in order]
    centers = [centers[i] for i in order]
    previews = []
    color_layers = []
    black_plates = []
    original_layers = []
    for i, (base_m, prod_m) in enumerate(zip(base_masks, prod_masks), start=1):
        rgb_color = lab_to_rgb_approx(*centers[i-1])
        preview_img = mask_to_rgba_image(prod_m, w, h, rgb_color, 220)
        layer_color_img = mask_to_rgba_image(prod_m, w, h, rgb_color, 255)
        plate_black_img = mask_to_rgba_image(prod_m, w, h, (0, 0, 0), 255)
        layer_original_img = mask_from_source(base_m, original, use_source_alpha=True)
        preview_name = f"preview_{i:02d}.png"
        layer_color_name = f"layer_color_{i:02d}.png"
        plate_black_name = f"plate_black_{i:02d}.png"
        layer_original_name = f"layer_original_{i:02d}.png"
        preview_img.save(os.path.join(run_dir, preview_name))
        layer_color_img.save(os.path.join(run_dir, layer_color_name))
        plate_black_img.save(os.path.join(run_dir, plate_black_name))
        layer_original_img.save(os.path.join(run_dir, layer_original_name))
        previews.append(preview_name)
        color_layers.append(layer_color_name)
        black_plates.append(plate_black_name)
        original_layers.append(layer_original_name)
    underbase_white_name = None
    underbase_black_name = None
    if include_underbase and prod_masks:
        ub_mask = build_underbase(prod_masks, underbase_spread_px)
        ub_white_img = mask_to_rgba_image(ub_mask, w, h, (255, 255, 255), 255)
        ub_black_img = mask_to_rgba_image(ub_mask, w, h, (0, 0, 0), 255)
        underbase_white_name = "underbase_white.png"
        underbase_black_name = "underbase_black.png"
        ub_white_img.save(os.path.join(run_dir, underbase_white_name))
        ub_black_img.save(os.path.join(run_dir, underbase_black_name))
    composite = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for name in original_layers:
        layer_img = Image.open(os.path.join(run_dir, name)).convert("RGBA")
        composite = Image.alpha_composite(composite, layer_img)
    composite_name = "composite.png"
    composite.save(os.path.join(run_dir, composite_name))
    zip_name = "plates.zip"
    zip_path = os.path.join(run_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in color_layers:
            zf.write(os.path.join(run_dir, name), arcname=os.path.join("layers_color", name))
        for name in black_plates:
            zf.write(os.path.join(run_dir, name), arcname=os.path.join("plates_black", name))
        for name in original_layers:
            zf.write(os.path.join(run_dir, name), arcname=os.path.join("layers_original", name))
        if underbase_white_name:
            zf.write(os.path.join(run_dir, underbase_white_name), arcname=os.path.join("underbase", underbase_white_name))
        if underbase_black_name:
            zf.write(os.path.join(run_dir, underbase_black_name), arcname=os.path.join("underbase", underbase_black_name))
        zf.write(os.path.join(run_dir, composite_name), arcname="composite.png")
        manifest = io.BytesIO()
        manifest.write(b"folders:\n")
        manifest.write(b"- layers_color: colorized layers for visual reference or DTF\n")
        manifest.write(b"- plates_black: black film positives for screen burning\n")
        manifest.write(b"- layers_original: original per-pixel color layers (composite matches source)\n")
        if include_underbase:
            manifest.write(b"- underbase: white/black underbase masks\n")
        manifest.write(b"files:\n- composite.png: composite of layers_original\n")
        zf.writestr("README.txt", manifest.getvalue())
    return render_template(
        "index.html",
        result={
            "run_id": run_id,
            "previews": previews,
            "plates": black_plates,
            "underbase": underbase_white_name or underbase_black_name,
            "zip_name": zip_name,
        },
    )


@app.route("/runs/<run_id>/<path:filename>")
def serve_run_file(run_id: str, filename: str):
    run_dir = os.path.join(RUNS_DIR, run_id)
    return send_from_directory(run_dir, filename, as_attachment=False)


@app.route("/download/<run_id>")
def download_zip(run_id: str):
    run_dir = os.path.join(RUNS_DIR, run_id)
    zip_path = os.path.join(run_dir, "plates.zip")
    if not os.path.exists(zip_path):
        return redirect(url_for("index"))
    return send_file(zip_path, as_attachment=True, download_name="plates.zip")

@app.route("/process-svg", methods=["POST"])
def process_svg():
    if "svg" not in request.files:
        return redirect(url_for("index"))
    file = request.files["svg"]
    if not file.filename.endswith(".svg"):
        return redirect(url_for("index"))

    run_id = str(uuid.uuid4())
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    svg_path = os.path.join(run_dir, "input.svg")
    file.save(svg_path)

    output_dir = os.path.join(run_dir, "svg_output")
    separate_color_layers(svg_path, output_dir)

    zip_path = os.path.join(run_dir, "svg_separated.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for fname in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, fname), arcname=fname)

    return send_file(zip_path, as_attachment=True, download_name="separated_svgs.zip")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
