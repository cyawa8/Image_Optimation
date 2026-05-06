import os
import sys

# ══════════════════════════════════════════════
#  STEP 1: ENV vars — harus paling awal
# ══════════════════════════════════════════════
os.environ['CUDA_VISIBLE_DEVICES']     = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ.pop('CUDA_HOME', None)
os.environ.pop('CUDA_PATH', None)

# ══════════════════════════════════════════════
#  STEP 2: Import torch lalu TIGA LAPIS PATCH
# ══════════════════════════════════════════════
import torch

# ── Patch A: is_available → selalu False ─────
# (Mencegah device selection ke CUDA)
torch.cuda.is_available = lambda: False

# ── Patch B: torch.load dengan map_location ──
# INI ADALAH ROOT CAUSE SEBENARNYA.
# simple-lama-inpainting memanggil torch.load(path) TANPA map_location.
# File big-lama.pt disimpan dengan CUDA tensor → langsung crash saat di-load
# di hardware non-CUDA (M2). Patch ini inject map_location='cpu' otomatis.
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, **kwargs):
    if map_location is None:
        map_location = 'cpu'   # paksa CPU jika tidak dispesifikasikan
    return _original_torch_load(f, map_location=map_location, **kwargs)

torch.load = _patched_torch_load

# ── Patch C: torch.device("cuda") → cpu ──────
# Mencegah kode internal yang hardcode string "cuda"
_original_torch_device = torch.device

class _SafeDevice(_original_torch_device):
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith('cuda'):
            args = ('cpu',) + args[1:]
        return _original_torch_device.__new__(cls, *args, **kwargs)

# Hanya patch string-based construction, tidak semua torch.device
# (supaya MPS tetap bisa dipakai)

# ══════════════════════════════════════════════
#  STEP 3: Tentukan device terbaik untuk M2
# ══════════════════════════════════════════════
def _best_device() -> torch.device:
    # Gunakan MPS jika tersedia (Apple Silicon GPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

TORCH_DEVICE = _best_device()
print(f"[Device] Menggunakan: {TORCH_DEVICE}")

# ══════════════════════════════════════════════
#  STEP 4: Bypass SSL untuk download model
# ══════════════════════════════════════════════
import ssl
if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

# ══════════════════════════════════════════════
#  STEP 5: Load SimpleLama SETELAH semua patch
# ══════════════════════════════════════════════
LAMA_AVAILABLE = False
lama_model     = None

try:
    from simple_lama_inpainting import SimpleLama

    # Coba inisialisasi — model weights akan di-load dengan map_location='cpu'
    # berkat Patch B di atas
    try:
        lama_model = SimpleLama(device=TORCH_DEVICE)
    except TypeError:
        # Versi lama SimpleLama tidak punya parameter device
        lama_model = SimpleLama()

    # Pastikan model benar-benar ada di device yang kita inginkan
    if hasattr(lama_model, 'model'):
        lama_model.model = lama_model.model.to(TORCH_DEVICE)
        param_devs = {str(p.device) for p in lama_model.model.parameters()}
        print(f"[LaMa] Model parameters device: {param_devs}")

    LAMA_AVAILABLE = True
    print(f"[LaMa] ✅ Berhasil dimuat di: {TORCH_DEVICE}")

except ImportError:
    print("[LaMa] ❌ Library tidak ditemukan.")
    print("         Install: pip install simple-lama-inpainting")
except Exception as e:
    LAMA_AVAILABLE = False
    print(f"[LaMa] ❌ Gagal: {type(e).__name__}: {e}")
    print("         Coba hapus cache model: rm -rf ~/.cache/torch/hub/checkpoints/big-lama.pt")

# ─────────────────────────────────────────────
#  Dependensi opsional
# ─────────────────────────────────────────────
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import numpy as np
import threading

# ══════════════════════════════════════════════
#  COLOUR / STYLE TOKENS
# ══════════════════════════════════════════════
BG        = "#0D0D0D"
SURFACE   = "#161616"
CARD      = "#1E1E1E"
BORDER    = "#2A2A2A"
ACCENT    = "#00E5FF"
ACCENT2   = "#FF3CAC"
TEXT      = "#F0F0F0"
MUTED     = "#666666"
SUCCESS   = "#00E676"
WARNING   = "#FFD600"
FONT_MAIN = ("Courier New", 10)
FONT_HEAD = ("Courier New", 13, "bold")
FONT_MONO = ("Courier New", 9)

# ══════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════
def pil_to_tk(img: Image.Image, max_w=380, max_h=280) -> ImageTk.PhotoImage:
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

def checkerboard(w, h, size=12):
    img  = Image.new("RGB", (w, h), "#2A2A2A")
    draw = ImageDraw.Draw(img)
    for y in range(0, h, size):
        for x in range(0, w, size):
            if (x // size + y // size) % 2 == 0:
                draw.rectangle([x, y, x + size - 1, y + size - 1], fill="#333333")
    return img

def composite_on_checker(rgba: Image.Image) -> Image.Image:
    bg = checkerboard(*rgba.size)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg

# ══════════════════════════════════════════════
#  PROCESSING FUNCTIONS
# ══════════════════════════════════════════════

def remove_logo_mask(img: Image.Image, mask_img: Image.Image) -> Image.Image:
    """Hapus logo/objek menggunakan AI LaMa inpainting."""
    mask_img = mask_img.resize(img.size, Image.NEAREST).convert("L")

    if LAMA_AVAILABLE:
        if CV2_AVAILABLE:
            cv_mask = np.array(mask_img)
            kernel  = np.ones((11, 11), np.uint8)
            cv_mask = cv2.dilate(cv_mask, kernel, iterations=1)
            mask_img = Image.fromarray(cv_mask)

        rgb_img    = img.convert("RGB")
        result_rgb = lama_model(rgb_img, mask_img)

        if img.mode == "RGBA":
            final_out = img.copy()
            final_out.paste(result_rgb, (0, 0))
            return final_out
        return result_rgb

    elif CV2_AVAILABLE:
        cv_img    = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv_mask   = np.array(mask_img)
        result_cv = cv2.inpaint(cv_img, cv_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        return Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))

    return img


def remove_watermark(img: Image.Image, threshold: int = 200, blend: float = 0.85) -> Image.Image:
    rgba = img.convert("RGBA")
    arr  = np.array(rgba, dtype=np.float32)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    bright  = (r > threshold) & (g > threshold) & (b > threshold)
    grey    = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (np.abs(r - b) < 25)
    wm_mask = bright & grey
    scale   = 1.0 - blend
    arr[wm_mask, 0] = arr[wm_mask, 0] * scale
    arr[wm_mask, 1] = arr[wm_mask, 1] * scale
    arr[wm_mask, 2] = arr[wm_mask, 2] * scale
    out = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return out.filter(ImageFilter.GaussianBlur(radius=1))


def remove_background(img: Image.Image) -> Image.Image:
    if REMBG_AVAILABLE:
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        result_bytes = rembg_remove(buf.read())
        return Image.open(BytesIO(result_bytes)).convert("RGBA")
    if CV2_AVAILABLE:
        return _bg_remove_grabcut(img)
    return _bg_remove_pil(img)

def _bg_remove_grabcut(img: Image.Image) -> Image.Image:
    bgr      = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask     = np.zeros(bgr.shape[:2], np.uint8)
    rect     = (5, 5, bgr.shape[1] - 10, bgr.shape[0] - 10)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    rgba = img.convert("RGBA")
    arr  = np.array(rgba)
    arr[..., 3] = fg_mask
    return Image.fromarray(arr)

def _bg_remove_pil(img: Image.Image, tolerance: int = 35) -> Image.Image:
    rgba    = img.convert("RGBA")
    arr     = np.array(rgba)
    h, w    = arr.shape[:2]
    corners = [arr[0, 0, :3], arr[0, w-1, :3], arr[h-1, 0, :3], arr[h-1, w-1, :3]]
    bg_col  = np.mean(corners, axis=0)
    diff    = np.abs(arr[..., :3].astype(int) - bg_col.astype(int))
    arr[diff.max(axis=2) < tolerance, 3] = 0
    return Image.fromarray(arr)

# ══════════════════════════════════════════════
#  GUI COMPONENTS
# ══════════════════════════════════════════════
class HeaderBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG, pady=12)
        tk.Label(self, text="[ IMAGE PROCESSOR AI ]",
                 font=("Courier New", 16, "bold"), bg=BG, fg=ACCENT).pack(side="left", padx=24)
        device_str = str(TORCH_DEVICE).upper() if LAMA_AVAILABLE else "—"
        ai_status  = f"Aktif [{device_str}]" if LAMA_AVAILABLE else "Nonaktif"
        tk.Label(self, text=f"v1.3 • LaMa AI: {ai_status}",
                 font=FONT_MONO, bg=BG,
                 fg=SUCCESS if LAMA_AVAILABLE else WARNING).pack(side="left")

class ImagePanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=CARD, padx=16, pady=16,
                         highlightbackground=BORDER, highlightthickness=1)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="INPUT", font=FONT_HEAD, bg=CARD, fg=ACCENT).pack(anchor="w")
        self.preview_lbl = tk.Label(self, bg="#111", width=42, height=16,
                                    relief="flat", cursor="hand2")
        self.preview_lbl.pack(pady=(10, 6))
        self.preview_lbl.bind("<Button-1>", lambda _: self.app.open_image())
        self.info_lbl = tk.Label(self, text="← klik untuk upload gambar",
                                 font=FONT_MONO, bg=CARD, fg=MUTED)
        self.info_lbl.pack()
        btn_frame = tk.Frame(self, bg=CARD)
        btn_frame.pack(pady=(10, 0))
        self._btn(btn_frame, "BUKA GAMBAR", self.app.open_image, ACCENT).pack(side="left", padx=4)
        self._btn(btn_frame, "RESET",        self.app.reset_image, MUTED).pack(side="left", padx=4)

    def _btn(self, parent, text, cmd, col):
        return tk.Button(parent, text=text, command=cmd, font=FONT_MONO,
                         bg=SURFACE, fg=col, activebackground=col, activeforeground=BG,
                         relief="flat", padx=10, pady=5, cursor="hand2",
                         highlightbackground=col, highlightthickness=1)

    def set_image(self, img: Image.Image, path=""):
        self._tk_img = pil_to_tk(img.copy())
        self.preview_lbl.configure(image=self._tk_img)
        name = os.path.basename(path) if path else f"{img.size[0]}×{img.size[1]}"
        self.info_lbl.configure(text=name, fg=TEXT)

class OutputPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=CARD, padx=16, pady=16,
                         highlightbackground=BORDER, highlightthickness=1)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="OUTPUT", font=FONT_HEAD, bg=CARD, fg=ACCENT2).pack(anchor="w")
        self.preview_lbl = tk.Label(self, bg="#111", width=42, height=16, relief="flat")
        self.preview_lbl.pack(pady=(10, 6))
        self.info_lbl = tk.Label(self, text="hasil muncul di sini",
                                 font=FONT_MONO, bg=CARD, fg=MUTED)
        self.info_lbl.pack()
        self.save_btn = tk.Button(self, text="SIMPAN HASIL", command=self.app.save_image,
                                  font=FONT_MONO, bg=SURFACE, fg=SUCCESS,
                                  activebackground=SUCCESS, activeforeground=BG,
                                  relief="flat", padx=10, pady=5, cursor="hand2",
                                  highlightbackground=SUCCESS, highlightthickness=1,
                                  state="disabled")
        self.save_btn.pack(pady=(10, 0))

    def set_image(self, img: Image.Image, label=""):
        display = composite_on_checker(img.convert("RGBA")) if img.mode == "RGBA" else img
        self._tk_img = pil_to_tk(display.copy())
        self.preview_lbl.configure(image=self._tk_img)
        self.info_lbl.configure(text=label or f"{img.size[0]}×{img.size[1]} {img.mode}", fg=TEXT)
        self.save_btn.configure(state="normal")

    def clear(self):
        self.preview_lbl.configure(image="")
        self.info_lbl.configure(text="hasil muncul di sini", fg=MUTED)
        self.save_btn.configure(state="disabled")

class ToolPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=SURFACE, padx=0, pady=0)
        self.app = app
        self._build()

    def _build(self):
        tab_bar = tk.Frame(self, bg=BG)
        tab_bar.pack(fill="x")
        self.tab_frames = {}
        self.tab_btns   = {}
        tabs = [
            ("✨  AI MAGIC ERASER",   "logo",      self._logo_content),
            ("💧  HAPUS WATERMARK",   "watermark", self._wm_content),
            ("✂  HAPUS BACKGROUND",  "bg",         self._bg_content),
        ]
        self.content_area = tk.Frame(self, bg=SURFACE, padx=20, pady=16)
        self.content_area.pack(fill="both", expand=True)
        for label, key, builder in tabs:
            btn = tk.Button(tab_bar, text=label, font=FONT_MONO,
                            bg=SURFACE, fg=MUTED, relief="flat",
                            padx=14, pady=8, cursor="hand2",
                            command=lambda k=key: self._switch(k))
            btn.pack(side="left")
            self.tab_btns[key] = btn
            frame = tk.Frame(self.content_area, bg=SURFACE)
            builder(frame)
            self.tab_frames[key] = frame
        self._switch("logo")

    def _switch(self, key):
        for k, f in self.tab_frames.items():
            f.pack_forget()
            self.tab_btns[k].configure(bg=SURFACE, fg=MUTED)
        self.tab_frames[key].pack(fill="both", expand=True)
        self.tab_btns[key].configure(bg=CARD, fg=ACCENT)

    def _logo_content(self, frame):
        tk.Label(frame,
                 text="Gunakan model AI (LaMa) untuk menghapus logo/objek "
                      "dan merekonstruksi background secara natural.",
                 font=FONT_MONO, bg=SURFACE, fg=MUTED,
                 wraplength=620, justify="left").pack(anchor="w", pady=(0, 10))
        self._action_btn(frame, "🖌️ BUKA KUAS PENGHAPUS",
                         self.app.start_brush_select, ACCENT2).pack(anchor="w")

    def _wm_content(self, frame):
        tk.Label(frame, text="Hapus watermark teks semi-transparan secara otomatis.",
                 font=FONT_MONO, bg=SURFACE, fg=MUTED,
                 wraplength=620, justify="left").pack(anchor="w")
        row = tk.Frame(frame, bg=SURFACE)
        row.pack(fill="x", pady=(12, 4))
        tk.Label(row, text="Threshold (brightness)", font=FONT_MONO,
                 bg=SURFACE, fg=TEXT).pack(side="left", padx=(0, 8))
        self.wm_threshold = tk.IntVar(value=190)
        tk.Scale(row, from_=100, to=255, orient="horizontal",
                 variable=self.wm_threshold, bg=SURFACE, fg=TEXT,
                 troughcolor=BORDER, highlightthickness=0,
                 length=200, font=FONT_MONO).pack(side="left")
        row2 = tk.Frame(frame, bg=SURFACE)
        row2.pack(fill="x", pady=(4, 12))
        tk.Label(row2, text="Blend strength       ", font=FONT_MONO,
                 bg=SURFACE, fg=TEXT).pack(side="left", padx=(0, 8))
        self.wm_blend = tk.DoubleVar(value=0.85)
        tk.Scale(row2, from_=0.1, to=1.0, resolution=0.05, orient="horizontal",
                 variable=self.wm_blend, bg=SURFACE, fg=TEXT,
                 troughcolor=BORDER, highlightthickness=0,
                 length=200, font=FONT_MONO).pack(side="left")
        self._action_btn(frame, "▶  HAPUS WATERMARK",
                         self.app.run_watermark, ACCENT).pack(anchor="w")

    def _bg_content(self, frame):
        method = ("rembg (AI)" if REMBG_AVAILABLE
                  else ("OpenCV GrabCut" if CV2_AVAILABLE else "PIL Fallback"))
        tk.Label(frame, text=f"Hapus background → transparan (PNG).  Metode aktif: {method}",
                 font=FONT_MONO, bg=SURFACE, fg=MUTED, wraplength=620).pack(anchor="w")
        self._action_btn(frame, "▶  HAPUS BACKGROUND",
                         self.app.run_bg_remove, ACCENT).pack(anchor="w", pady=(10, 0))

    def _action_btn(self, parent, text, cmd, col):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Courier New", 10, "bold"),
                         bg=col, fg=BG, activebackground=TEXT, activeforeground=BG,
                         relief="flat", padx=14, pady=7, cursor="hand2")

class StatusBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG, pady=6)
        self.lbl = tk.Label(self, text="siap.", font=FONT_MONO, bg=BG, fg=MUTED, anchor="w")
        self.lbl.pack(side="left", padx=16)
        self.progress = ttk.Progressbar(self, length=220, mode="indeterminate")
        self.progress.pack(side="right", padx=16)

    def set(self, msg, colour=MUTED):
        self.lbl.configure(text=msg, fg=colour)
        self.lbl.update_idletasks()

    def busy(self, on=True):
        if on: self.progress.start(15)
        else:  self.progress.stop()

# ══════════════════════════════════════════════
#  BRUSH SELECTOR OVERLAY
# ══════════════════════════════════════════════
class BrushSelector(tk.Toplevel):
    def __init__(self, parent, img: Image.Image, callback):
        super().__init__(parent, bg=BG)
        self.title("✨ AI Magic Eraser - Kuas")
        self.callback = callback

        max_w, max_h = 1000, 700
        self.display_img = img.copy()
        self.display_img.thumbnail((max_w, max_h), Image.LANCZOS)

        self.mask_img  = Image.new("L", self.display_img.size, 0)
        self.mask_draw = ImageDraw.Draw(self.mask_img)

        toolbar = tk.Frame(self, bg=SURFACE, pady=10, padx=10)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="Ukuran Kuas:", bg=SURFACE, fg=TEXT,
                 font=FONT_MAIN).pack(side="left")
        self.brush_size = tk.Scale(toolbar, from_=5, to=80, orient="horizontal",
                                   bg=SURFACE, fg=ACCENT, highlightthickness=0,
                                   length=200, troughcolor=BORDER)
        self.brush_size.set(20)
        self.brush_size.pack(side="left", padx=10)
        tk.Button(toolbar, text="✨ Generate AI", command=self.apply,
                  bg=ACCENT2, fg=BG, font=FONT_HEAD, cursor="hand2",
                  relief="flat", padx=10).pack(side="right", padx=5)
        tk.Button(toolbar, text="Reset Kuas", command=self.clear_brush,
                  bg=BORDER, fg=TEXT, font=FONT_MONO, cursor="hand2",
                  relief="flat", padx=10).pack(side="right", padx=5)

        tk.Label(self,
                 text="Coret area yang ingin dihapus. Pastikan seluruh bagian tertutup warna merah.",
                 bg=BG, fg=WARNING, font=FONT_MONO, pady=5).pack()

        self.tk_img = ImageTk.PhotoImage(self.display_img)
        self.canvas = tk.Canvas(self, width=self.display_img.width,
                                height=self.display_img.height,
                                cursor="circle", bg=BG, highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.last_x = self.last_y = None
        self.canvas.bind("<Button-1>",        self.start_draw)
        self.canvas.bind("<B1-Motion>",       self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.last_x is not None:
            r = self.brush_size.get()
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill="#FF3CAC", width=r,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="brush")
            self.mask_draw.line([self.last_x, self.last_y, x, y],
                                fill=255, width=r, joint="curve")
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.last_x = self.last_y = None

    def clear_brush(self):
        self.canvas.delete("brush")
        self.mask_img  = Image.new("L", self.display_img.size, 0)
        self.mask_draw = ImageDraw.Draw(self.mask_img)

    def apply(self):
        self.callback(self.mask_img)
        self.destroy()

# ══════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processor AI")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(760, 560)
        self._src_img:    Image.Image | None = None
        self._src_path:   str = ""
        self._result_img: Image.Image | None = None
        self._build()
        self._center_window(1060, 640)

    def _center_window(self, w, h):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build(self):
        ttk.Style().configure("TProgressbar", troughcolor=BORDER,
                              background=ACCENT, thickness=4)
        HeaderBar(self).pack(fill="x")
        preview_row = tk.Frame(self, bg=BG)
        preview_row.pack(fill="both", expand=False, padx=16, pady=(0, 8))
        self.input_panel  = ImagePanel(preview_row, self)
        self.output_panel = OutputPanel(preview_row, self)
        self.input_panel.pack(side="left",  fill="both", expand=True, padx=(0, 6))
        self.output_panel.pack(side="right", fill="both", expand=True, padx=(6, 0))
        self.tools = ToolPanel(self, self)
        self.tools.pack(fill="x", padx=16, pady=(0, 8))
        self.status = StatusBar(self)
        self.status.pack(fill="x")

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"), ("All", "*.*")])
        if not path: return
        try:
            self._src_img  = Image.open(path).convert("RGBA")
            self._src_path = path
            self.input_panel.set_image(self._src_img, path)
            self.output_panel.clear()
            self._result_img = None
            self.status.set(f"Dibuka: {os.path.basename(path)}", SUCCESS)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_image(self):
        self._src_img = self._result_img = None
        self._src_path = ""
        self.input_panel.preview_lbl.configure(image="")
        self.input_panel.info_lbl.configure(text="← klik untuk upload gambar", fg=MUTED)
        self.output_panel.clear()
        self.status.set("reset.", MUTED)

    def save_image(self):
        if self._result_img is None: return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")])
        if not path: return
        try:
            save_img = self._result_img
            if path.lower().endswith((".jpg", ".jpeg")) and save_img.mode == "RGBA":
                save_img = save_img.convert("RGB")
            save_img.save(path)
            self.status.set(f"Disimpan: {os.path.basename(path)}", SUCCESS)
        except Exception as e:
            messagebox.showerror("Gagal simpan", str(e))

    def start_brush_select(self):
        if self._src_img is None:
            messagebox.showwarning("Peringatan", "Upload gambar terlebih dahulu.")
            return
        BrushSelector(self, self._src_img.copy(), self._on_brush_applied)

    def _on_brush_applied(self, mask_img):
        self._run(lambda img: remove_logo_mask(img, mask_img), "Proses AI LaMa")

    def _run(self, fn, label):
        if self._src_img is None:
            messagebox.showwarning("Peringatan", "Upload gambar terlebih dahulu.")
            return
        def worker():
            self.status.set(f"{label} sedang berjalan... ⏳", WARNING)
            self.status.busy(True)
            try:
                result = fn(self._src_img.copy())
                self._result_img = result
                self.output_panel.set_image(result, label)
                self.status.set(f"Selesai: {label} ✨", SUCCESS)
            except Exception as e:
                self.status.set(f"Error: {e}", ACCENT2)
                messagebox.showerror("Error", str(e))
            finally:
                self.status.busy(False)
        threading.Thread(target=worker, daemon=True).start()

    def run_watermark(self):
        thr   = self.tools.wm_threshold.get()
        blend = self.tools.wm_blend.get()
        self._run(lambda img: remove_watermark(img, thr, blend), "Hapus Watermark")

    def run_bg_remove(self):
        self._run(remove_background, "Hapus Background")


if __name__ == "__main__":
    app = App()
    app.mainloop()