import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import numpy as np
import os
import threading

# ─────────────────────────────────────────────
#  Try importing optional heavy deps (AI Models)
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

# Import AI LaMa Inpainting
# --- Tambahkan/Perbaiki di bagian atas file ---
try:
    # Menggunakan nama modul yang sesuai dengan folder di site-packages Anda
    from simple_lama_inpainting import SimpleLama 
    lama_model = SimpleLama()
    LAMA_AVAILABLE = True
    print("AI LaMa Berhasil Dimuat! ✨")
except Exception as e:
    LAMA_AVAILABLE = False
    print(f"AI Gagal Dimuat. Detail Error: {e}")
    print("Gunakan perintah: python -m pip install simple-lama-inpainting")

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
    img = Image.new("RGB", (w, h), "#2A2A2A")
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

# ── 1. REMOVE LOGO (Menggunakan Mask Kuas & AI LaMa Deep Learning) ──
def remove_logo_mask(img: Image.Image, mask_img: Image.Image) -> Image.Image:
    """Menghapus logo dengan AI LaMa untuk hasil paling natural."""
    # Pastikan mask sesuai ukuran asli gambar
    mask_img = mask_img.resize(img.size, Image.NEAREST).convert("L")
    
    if LAMA_AVAILABLE:
        # PENTING: Pertebal mask sedikit agar AI bisa menghapus sisa bayangan/tepi
        if CV2_AVAILABLE:
            cv_mask = np.array(mask_img)
            kernel = np.ones((11, 11), np.uint8) # Ketebalan ekstra 11px
            cv_mask = cv2.dilate(cv_mask, kernel, iterations=1)
            mask_img = Image.fromarray(cv_mask)

        # Konversi ke RGB karena AI tidak mendukung RGBA secara langsung
        rgb_img = img.convert("RGB")
        
        # Proses AI Inpainting
        result_rgb = lama_model(rgb_img, mask_img)
        
        # Jika gambar asli punya transparansi, tempelkan kembali hasil AI-nya
        if img.mode == "RGBA":
            final_out = img.copy()
            final_out.paste(result_rgb, (0, 0))
            return final_out
        return result_rgb
    
    # Jika AI gagal, gunakan OpenCV Inpainting sebagai cadangan terakhir
    elif CV2_AVAILABLE:
        cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv_mask = np.array(mask_img)
        result_cv = cv2.inpaint(cv_img, cv_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        return Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
    
    return img

# ── 2. REMOVE WATERMARK ──
def remove_watermark(img: Image.Image, threshold: int = 200, blend: float = 0.85) -> Image.Image:
    rgba = img.convert("RGBA")
    arr  = np.array(rgba, dtype=np.float32)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    bright  = (r > threshold) & (g > threshold) & (b > threshold)
    grey    = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (np.abs(r - b) < 25)
    wm_mask = bright & grey
    scale = 1.0 - blend
    arr[wm_mask, 0] = arr[wm_mask, 0] * scale
    arr[wm_mask, 1] = arr[wm_mask, 1] * scale
    arr[wm_mask, 2] = arr[wm_mask, 2] * scale
    out = Image.fromarray(arr.astype(np.uint8), "RGBA")
    out = out.filter(ImageFilter.GaussianBlur(radius=1))
    return out

# ── 3. REMOVE BACKGROUND ──
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
    bgr  = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    rect = (5, 5, bgr.shape[1] - 10, bgr.shape[0] - 10)
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
    rgba = img.convert("RGBA")
    arr  = np.array(rgba)
    h, w = arr.shape[:2]
    corners = [arr[0, 0, :3], arr[0, w-1, :3], arr[h-1, 0, :3], arr[h-1, w-1, :3]]
    bg_col  = np.mean(corners, axis=0)
    diff    = np.abs(arr[..., :3].astype(int) - bg_col.astype(int))
    is_bg   = diff.max(axis=2) < tolerance
    arr[is_bg, 3] = 0
    return Image.fromarray(arr)

# ══════════════════════════════════════════════
#  GUI COMPONENTS
# ══════════════════════════════════════════════
class HeaderBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG, pady=12)
        tk.Label(self, text="[ IMAGE PROCESSOR AI ]", font=("Courier New", 16, "bold"),
                 bg=BG, fg=ACCENT).pack(side="left", padx=24)
        ai_status = "Aktif" if LAMA_AVAILABLE else "Nonaktif"
        tk.Label(self, text=f"v1.1 • Deep Learning AI: {ai_status}",
                 font=FONT_MONO, bg=BG, fg=SUCCESS if LAMA_AVAILABLE else WARNING).pack(side="left")

class ImagePanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=CARD, padx=16, pady=16, highlightbackground=BORDER, highlightthickness=1)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="INPUT", font=FONT_HEAD, bg=CARD, fg=ACCENT).pack(anchor="w")
        self.preview_lbl = tk.Label(self, bg="#111", width=42, height=16, relief="flat", cursor="hand2")
        self.preview_lbl.pack(pady=(10, 6))
        self.preview_lbl.bind("<Button-1>", lambda _: self.app.open_image())
        self.info_lbl = tk.Label(self, text="← klik untuk upload gambar", font=FONT_MONO, bg=CARD, fg=MUTED)
        self.info_lbl.pack()
        btn_frame = tk.Frame(self, bg=CARD)
        btn_frame.pack(pady=(10, 0))
        self._btn(btn_frame, "BUKA GAMBAR", self.app.open_image, ACCENT).pack(side="left", padx=4)
        self._btn(btn_frame, "RESET",        self.app.reset_image, MUTED).pack(side="left", padx=4)

    def _btn(self, parent, text, cmd, col):
        return tk.Button(parent, text=text, command=cmd, font=FONT_MONO, bg=SURFACE, fg=col,
                         activebackground=col, activeforeground=BG, relief="flat", padx=10, pady=5, cursor="hand2",
                         highlightbackground=col, highlightthickness=1)

    def set_image(self, img: Image.Image, path=""):
        self._tk_img = pil_to_tk(img.copy())
        self.preview_lbl.configure(image=self._tk_img)
        name = os.path.basename(path) if path else f"{img.size[0]}×{img.size[1]}"
        self.info_lbl.configure(text=name, fg=TEXT)

class OutputPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=CARD, padx=16, pady=16, highlightbackground=BORDER, highlightthickness=1)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="OUTPUT", font=FONT_HEAD, bg=CARD, fg=ACCENT2).pack(anchor="w")
        self.preview_lbl = tk.Label(self, bg="#111", width=42, height=16, relief="flat")
        self.preview_lbl.pack(pady=(10, 6))
        self.info_lbl = tk.Label(self, text="hasil muncul di sini", font=FONT_MONO, bg=CARD, fg=MUTED)
        self.info_lbl.pack()
        self.save_btn = tk.Button(self, text="SIMPAN HASIL", command=self.app.save_image,
                                  font=FONT_MONO, bg=SURFACE, fg=SUCCESS, activebackground=SUCCESS, activeforeground=BG,
                                  relief="flat", padx=10, pady=5, cursor="hand2",
                                  highlightbackground=SUCCESS, highlightthickness=1, state="disabled")
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
            ("✨  AI MAGIC ERASER", "logo",       self._logo_content),
            ("💧  HAPUS WATERMARK",    "watermark",  self._wm_content),
            ("✂  HAPUS BACKGROUND",   "bg",          self._bg_content),
        ]
        self.content_area = tk.Frame(self, bg=SURFACE, padx=20, pady=16)
        self.content_area.pack(fill="both", expand=True)

        for label, key, builder in tabs:
            btn = tk.Button(tab_bar, text=label, font=FONT_MONO, bg=SURFACE, fg=MUTED, relief="flat",
                            padx=14, pady=8, cursor="hand2", command=lambda k=key: self._switch(k))
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
        tk.Label(frame, text="Gunakan model Deep Learning AI (LaMa) untuk menghapus objek/logo dan menggenerate ulang background secara natural.",
                 font=FONT_MONO, bg=SURFACE, fg=MUTED, wraplength=620, justify="left").pack(anchor="w", pady=(0, 10))
        btn_row = tk.Frame(frame, bg=SURFACE)
        btn_row.pack(anchor="w")
        self._action_btn(btn_row, "🖌️ BUKA KUAS PENGHAPUS", self.app.start_brush_select, ACCENT2).pack(side="left")

    def _wm_content(self, frame):
        tk.Label(frame, text="Hapus watermark teks semi-transparan (putih/abu-abu) secara otomatis.",
                 font=FONT_MONO, bg=SURFACE, fg=MUTED, wraplength=620, justify="left").pack(anchor="w")
        row = tk.Frame(frame, bg=SURFACE)
        row.pack(fill="x", pady=(12, 4))
        tk.Label(row, text="Threshold (brightness)", font=FONT_MONO, bg=SURFACE, fg=TEXT).pack(side="left", padx=(0,8))
        self.wm_threshold = tk.IntVar(value=190)
        tk.Scale(row, from_=100, to=255, orient="horizontal", variable=self.wm_threshold,
                 bg=SURFACE, fg=TEXT, troughcolor=BORDER, highlightthickness=0, length=200, font=FONT_MONO).pack(side="left")
        row2 = tk.Frame(frame, bg=SURFACE)
        row2.pack(fill="x", pady=(4, 12))
        tk.Label(row2, text="Blend strength       ", font=FONT_MONO, bg=SURFACE, fg=TEXT).pack(side="left", padx=(0,8))
        self.wm_blend = tk.DoubleVar(value=0.85)
        tk.Scale(row2, from_=0.1, to=1.0, resolution=0.05, orient="horizontal", variable=self.wm_blend,
                 bg=SURFACE, fg=TEXT, troughcolor=BORDER, highlightthickness=0, length=200, font=FONT_MONO).pack(side="left")
        self._action_btn(frame, "▶  HAPUS WATERMARK", self.app.run_watermark, ACCENT).pack(anchor="w")

    def _bg_content(self, frame):
        method = "rembg (AI)" if REMBG_AVAILABLE else ("OpenCV GrabCut" if CV2_AVAILABLE else "PIL Fallback")
        tk.Label(frame, text=f"Hapus background → transparan (PNG).  Metode aktif: {method}",
                 font=FONT_MONO, bg=SURFACE, fg=MUTED, wraplength=620).pack(anchor="w")
        self._action_btn(frame, "▶  HAPUS BACKGROUND", self.app.run_bg_remove, ACCENT).pack(anchor="w", pady=(10, 0))

    def _action_btn(self, parent, text, cmd, col):
        return tk.Button(parent, text=text, command=cmd, font=("Courier New", 10, "bold"),
                         bg=col, fg=BG, activebackground=TEXT, activeforeground=BG, relief="flat", padx=14, pady=7, cursor="hand2")

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
        else: self.progress.stop()

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
        
        self.mask_img = Image.new("L", self.display_img.size, 0)
        self.mask_draw = ImageDraw.Draw(self.mask_img)

        toolbar = tk.Frame(self, bg=SURFACE, pady=10, padx=10)
        toolbar.pack(fill="x")
        
        tk.Label(toolbar, text="Ukuran Kuas:", bg=SURFACE, fg=TEXT, font=FONT_MAIN).pack(side="left")
        self.brush_size = tk.Scale(toolbar, from_=5, to=80, orient="horizontal", 
                                   bg=SURFACE, fg=ACCENT, highlightthickness=0, length=200, troughcolor=BORDER)
        self.brush_size.set(20)
        self.brush_size.pack(side="left", padx=10)
        
        btn_apply = tk.Button(toolbar, text="✨ Generate AI", command=self.apply, 
                              bg=ACCENT2, fg=BG, font=FONT_HEAD, cursor="hand2", relief="flat", padx=10)
        btn_apply.pack(side="right", padx=5)
        
        btn_reset = tk.Button(toolbar, text="Reset Kuas", command=self.clear_brush, 
                              bg=BORDER, fg=TEXT, font=FONT_MONO, cursor="hand2", relief="flat", padx=10)
        btn_reset.pack(side="right", padx=5)
        
        tk.Label(self, text="Coret pada logo/area yang ingin dihapus. Pastikan seluruh bagian yang tidak diinginkan tertutup warna merah.", 
                 bg=BG, fg=WARNING, font=FONT_MONO, pady=5).pack()

        self.tk_img = ImageTk.PhotoImage(self.display_img)
        self.canvas = tk.Canvas(self, width=self.display_img.width, height=self.display_img.height, 
                                cursor="circle", bg=BG, highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.last_x, self.last_y = None, None
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.last_x and self.last_y:
            r = self.brush_size.get()
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                    fill="#FF3CAC", width=r, capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="brush")
            self.mask_draw.line([self.last_x, self.last_y, x, y], 
                                fill=255, width=r, joint="curve")
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.last_x, self.last_y = None, None

    def clear_brush(self):
        self.canvas.delete("brush")
        self.mask_img = Image.new("L", self.display_img.size, 0)
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
        self._src_img: Image.Image | None = None
        self._src_path: str = ""
        self._result_img: Image.Image | None = None
        self._build()
        self._center_window(1060, 640)

    def _center_window(self, w, h):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build(self):
        ttk.Style().configure("TProgressbar", troughcolor=BORDER, background=ACCENT, thickness=4)
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
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"), ("All", "*.*")])
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
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")])
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
            self.status.set(f"{label} sedang berjalan, harap tunggu... ⏳", WARNING)
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