"""
proses_Gambar_FINAL.py
======================
Image Processor AI  —  Logo Removal · Watermark Removal · Background Removal
Fix: Custom LaMa loader yang bypass simple-lama-inpainting library sepenuhnya.
"""
import os
import sys
import ssl
import urllib.request
import hashlib
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — ENV vars SEBELUM torch diimport (wajib, jangan dipindah)
# ══════════════════════════════════════════════════════════════════════
os.environ['CUDA_VISIBLE_DEVICES']        = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
for _k in ('CUDA_HOME', 'CUDA_PATH', 'FORCE_CUDA'):
    os.environ.pop(_k, None)

# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — Import torch dan pasang patch map_location
# ══════════════════════════════════════════════════════════════════════
import torch

# Patch torch.load: inject map_location='cpu' jika tidak dispesifikasikan.
# Ini menangani kasus di mana library internal memanggil torch.load(path)
# tanpa map_location, yang menyebabkan CUDA init pada mesin non-CUDA.
_orig_torch_load = torch.load
def _safe_torch_load(f, map_location=None, **kw):
    if map_location is None:
        map_location = 'cpu'
    return _orig_torch_load(f, map_location=map_location, **kw)
torch.load = _safe_torch_load

# Patch torch.jit.load juga (LaMa menggunakan TorchScript)
_orig_jit_load = torch.jit.load
def _safe_jit_load(f, map_location=None, **kw):
    if map_location is None:
        map_location = 'cpu'
    return _orig_jit_load(f, map_location=map_location, **kw)
torch.jit.load = _safe_jit_load

# Killswitch CUDA di Python layer (defence in depth)
torch.cuda.is_available  = lambda: False
torch.cuda.device_count  = lambda: 0

# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — Pilih device terbaik
# ══════════════════════════════════════════════════════════════════════
def _pick_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')

TORCH_DEVICE = _pick_device()
print(f'[Device] Menggunakan: {TORCH_DEVICE}')

# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — Custom LaMa Loader
#           Bypass simple-lama-inpainting sepenuhnya.
#           Library itu menggunakan torch.jit.load dengan device info
#           yang tertanam di TorchScript IR → tidak bisa di-override
#           dari luar. Kita load model sendiri dengan kontrol penuh.
# ══════════════════════════════════════════════════════════════════════

# Lokasi cache model (sama dengan yang dipakai simple-lama-inpainting)
_MODEL_CACHE = Path.home() / '.cache' / 'simple-lama-inpainting'
_MODEL_PATH  = _MODEL_CACHE / 'big-lama.pt'
_MODEL_URL   = ('https://github.com/enesmsahin/simple-lama-inpainting'
                '/releases/download/v0.1.0/big-lama.pt')

def _download_model() -> bool:
    """Download big-lama.pt jika belum ada."""
    if _MODEL_PATH.exists():
        return True
    print('[LaMa] Model tidak ditemukan, mengunduh big-lama.pt (~200 MB)...')
    _MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    ssl_ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(_MODEL_URL, context=ssl_ctx) as resp, \
             open(_MODEL_PATH, 'wb') as fout:
            total = int(resp.headers.get('Content-Length', 0))
            done  = 0
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                fout.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f'\r[LaMa] Download: {pct}%  ', end='', flush=True)
        print('\n[LaMa] Download selesai.')
        return True
    except Exception as e:
        print(f'\n[LaMa] Download gagal: {e}')
        if _MODEL_PATH.exists():
            _MODEL_PATH.unlink()
        return False


class _LamaModel:
    """
    Custom wrapper LaMa yang menggunakan torch.jit.load dengan
    map_location='cpu', lalu memindahkan model ke TORCH_DEVICE.
    Ini menghindari bug di simple-lama-inpainting yang hardcode CUDA.
    """
    def __init__(self, device: torch.device):
        self.device = device
        # Load model weight ke CPU dulu, lalu pindah ke device target.
        # map_location='cpu' mencegah error aten::empty_strided CUDA.
        self.model = torch.jit.load(str(_MODEL_PATH), map_location='cpu')
        self.model = self.model.to(device)
        self.model.eval()
        print(f'[LaMa] Model dimuat di device: {device}')

    @torch.no_grad()
    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        # 1. Preprocessing (Tetap sama)
        mask = mask.resize(image.size, Image.NEAREST).convert('L')
        
        # Konversi ke Tensor dan pindahkan ke device (MPS/CPU)
        img_t = torch.from_numpy(np.array(image.convert('RGB')).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy((np.array(mask) > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 2. PERBAIKAN KRITIS: Kirim sebagai argumen terpisah, bukan Dictionary
        # Berdasarkan error: forward(Tensor image, Tensor mask) -> Tensor
        result = self.model(img_t, mask_t)

        # 3. PERBAIKAN OUTPUT: Model mengembalikan Tensor secara langsung, bukan dict
        # Jika hasil adalah Tensor tunggal [1, 3, H, W]
        if isinstance(result, torch.Tensor):
            out_t = result[0]
        else:
            # Jika model versi lama mengembalikan dict (fallback)
            out_t = result['inpainted'][0]

        # Konversi kembali ke PIL Image
        out_np = (out_t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out_np)


LAMA_AVAILABLE = False
lama_model     = None

if _download_model():
    try:
        lama_model     = _LamaModel(TORCH_DEVICE)
        LAMA_AVAILABLE = True
        print(f'[LaMa] ✅ Siap digunakan pada {TORCH_DEVICE}')
    except Exception as e:
        print(f'[LaMa] ❌ Gagal load model: {type(e).__name__}: {e}')
        print('[LaMa]    Coba hapus cache dan download ulang:')
        print(f'[LaMa]    rm {_MODEL_PATH}')
else:
    print('[LaMa] ❌ Model tidak tersedia.')

# ──────────────────────────────────────────────
#  Dependensi opsional
# ──────────────────────────────────────────────
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
BG        = '#0D0D0D'
SURFACE   = '#161616'
CARD      = '#1E1E1E'
BORDER    = '#2A2A2A'
ACCENT    = '#00E5FF'
ACCENT2   = '#FF3CAC'
TEXT      = '#F0F0F0'
MUTED     = '#666666'
SUCCESS   = '#00E676'
WARNING   = '#FFD600'
FONT_MAIN = ('Courier New', 10)
FONT_HEAD = ('Courier New', 13, 'bold')
FONT_MONO = ('Courier New', 9)

# ══════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════
def pil_to_tk(img: Image.Image, max_w=380, max_h=280) -> ImageTk.PhotoImage:
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

def checkerboard(w, h, size=12):
    img  = Image.new('RGB', (w, h), '#2A2A2A')
    draw = ImageDraw.Draw(img)
    for y in range(0, h, size):
        for x in range(0, w, size):
            if (x // size + y // size) % 2 == 0:
                draw.rectangle([x, y, x+size-1, y+size-1], fill='#333333')
    return img

def composite_on_checker(rgba: Image.Image) -> Image.Image:
    bg = checkerboard(*rgba.size)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg

# ══════════════════════════════════════════════
#  PROCESSING FUNCTIONS
# ══════════════════════════════════════════════
def remove_logo_mask(img: Image.Image, mask_img: Image.Image) -> Image.Image:
    mask_img = mask_img.resize(img.size, Image.NEAREST).convert('L')

    if LAMA_AVAILABLE:
        if CV2_AVAILABLE:
            cv_mask = np.array(mask_img)
            kernel  = np.ones((11, 11), np.uint8)
            cv_mask = cv2.dilate(cv_mask, kernel, iterations=1)
            mask_img = Image.fromarray(cv_mask)
        rgb_img    = img.convert('RGB')
        result_rgb = lama_model(rgb_img, mask_img)
        if img.mode == 'RGBA':
            out = img.copy()
            out.paste(result_rgb, (0, 0))
            return out
        return result_rgb

    elif CV2_AVAILABLE:
        cv_img    = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        cv_mask   = np.array(mask_img)
        result_cv = cv2.inpaint(cv_img, cv_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        return Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))

    return img


def remove_watermark(img: Image.Image, threshold: int = 200, blend: float = 0.85) -> Image.Image:
    rgba = img.convert('RGBA')
    arr  = np.array(rgba, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    bright  = (r > threshold) & (g > threshold) & (b > threshold)
    grey    = (np.abs(r-g) < 25) & (np.abs(g-b) < 25) & (np.abs(r-b) < 25)
    wm_mask = bright & grey
    scale   = 1.0 - blend
    arr[wm_mask, 0] *= scale
    arr[wm_mask, 1] *= scale
    arr[wm_mask, 2] *= scale
    return Image.fromarray(arr.astype(np.uint8), 'RGBA').filter(ImageFilter.GaussianBlur(1))


def remove_background(img: Image.Image) -> Image.Image:
    if REMBG_AVAILABLE:
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return Image.open(BytesIO(rembg_remove(buf.read()))).convert('RGBA')
    if CV2_AVAILABLE:
        return _bg_grabcut(img)
    return _bg_pil(img)

def _bg_grabcut(img: Image.Image) -> Image.Image:
    bgr      = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    mask     = np.zeros(bgr.shape[:2], np.uint8)
    rect     = (5, 5, bgr.shape[1]-10, bgr.shape[0]-10)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    fg = cv2.GaussianBlur(fg, (5, 5), 0)
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    arr = np.array(img.convert('RGBA'))
    arr[..., 3] = fg
    return Image.fromarray(arr)

def _bg_pil(img: Image.Image, tol: int = 35) -> Image.Image:
    arr     = np.array(img.convert('RGBA'))
    h, w    = arr.shape[:2]
    corners = [arr[0,0,:3], arr[0,w-1,:3], arr[h-1,0,:3], arr[h-1,w-1,:3]]
    bg_col  = np.mean(corners, axis=0)
    diff    = np.abs(arr[..., :3].astype(int) - bg_col.astype(int))
    arr[diff.max(axis=2) < tol, 3] = 0
    return Image.fromarray(arr)

# ══════════════════════════════════════════════
#  GUI COMPONENTS
# ══════════════════════════════════════════════
class HeaderBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG, pady=12)
        tk.Label(self, text='[ IMAGE PROCESSOR AI ]',
                 font=('Courier New', 16, 'bold'), bg=BG, fg=ACCENT).pack(side='left', padx=24)
        dev_str   = str(TORCH_DEVICE).upper() if LAMA_AVAILABLE else '—'
        ai_status = f'Aktif [{dev_str}]' if LAMA_AVAILABLE else 'Nonaktif'
        tk.Label(self, text=f'v2.0 • LaMa AI: {ai_status}', font=FONT_MONO, bg=BG,
                 fg=SUCCESS if LAMA_AVAILABLE else WARNING).pack(side='left')


class ImagePanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=CARD, padx=16, pady=16,
                         highlightbackground=BORDER, highlightthickness=1)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text='INPUT', font=FONT_HEAD, bg=CARD, fg=ACCENT).pack(anchor='w')
        self.preview_lbl = tk.Label(self, bg='#111', width=42, height=16,
                                    relief='flat', cursor='hand2')
        self.preview_lbl.pack(pady=(10, 6))
        self.preview_lbl.bind('<Button-1>', lambda _: self.app.open_image())
        self.info_lbl = tk.Label(self, text='← klik untuk upload gambar',
                                 font=FONT_MONO, bg=CARD, fg=MUTED)
        self.info_lbl.pack()
        row = tk.Frame(self, bg=CARD)
        row.pack(pady=(10, 0))
        self._btn(row, 'BUKA GAMBAR', self.app.open_image, ACCENT).pack(side='left', padx=4)
        self._btn(row, 'RESET',        self.app.reset_image, MUTED).pack(side='left', padx=4)

    def _btn(self, p, t, c, col):
        return tk.Button(p, text=t, command=c, font=FONT_MONO, bg=SURFACE, fg=col,
                         activebackground=col, activeforeground=BG, relief='flat',
                         padx=10, pady=5, cursor='hand2',
                         highlightbackground=col, highlightthickness=1)

    def set_image(self, img, path=''):
        self._tk = pil_to_tk(img.copy())
        self.preview_lbl.configure(image=self._tk)
        self.info_lbl.configure(
            text=os.path.basename(path) if path else f'{img.size[0]}×{img.size[1]}', fg=TEXT)


class OutputPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=CARD, padx=16, pady=16,
                         highlightbackground=BORDER, highlightthickness=1)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text='OUTPUT', font=FONT_HEAD, bg=CARD, fg=ACCENT2).pack(anchor='w')
        self.preview_lbl = tk.Label(self, bg='#111', width=42, height=16, relief='flat')
        self.preview_lbl.pack(pady=(10, 6))
        self.info_lbl = tk.Label(self, text='hasil muncul di sini',
                                 font=FONT_MONO, bg=CARD, fg=MUTED)
        self.info_lbl.pack()
        self.save_btn = tk.Button(self, text='SIMPAN HASIL', command=self.app.save_image,
                                  font=FONT_MONO, bg=SURFACE, fg=SUCCESS,
                                  activebackground=SUCCESS, activeforeground=BG,
                                  relief='flat', padx=10, pady=5, cursor='hand2',
                                  highlightbackground=SUCCESS, highlightthickness=1,
                                  state='disabled')
        self.save_btn.pack(pady=(10, 0))

    def set_image(self, img, label=''):
        display  = composite_on_checker(img.convert('RGBA')) if img.mode == 'RGBA' else img
        self._tk = pil_to_tk(display.copy())
        self.preview_lbl.configure(image=self._tk)
        self.info_lbl.configure(text=label or f'{img.size[0]}×{img.size[1]} {img.mode}', fg=TEXT)
        self.save_btn.configure(state='normal')

    def clear(self):
        self.preview_lbl.configure(image='')
        self.info_lbl.configure(text='hasil muncul di sini', fg=MUTED)
        self.save_btn.configure(state='disabled')


class ToolPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=SURFACE)
        self.app = app
        self._build()

    def _build(self):
        tab_bar = tk.Frame(self, bg=BG)
        tab_bar.pack(fill='x')
        self.tab_frames, self.tab_btns = {}, {}
        tabs = [
            ('✨  AI MAGIC ERASER',   'logo',      self._logo_tab),
            ('💧  HAPUS WATERMARK',   'watermark', self._wm_tab),
            ('✂  HAPUS BACKGROUND',  'bg',         self._bg_tab),
        ]
        self.content_area = tk.Frame(self, bg=SURFACE, padx=20, pady=16)
        self.content_area.pack(fill='both', expand=True)
        for label, key, builder in tabs:
            btn = tk.Button(tab_bar, text=label, font=FONT_MONO, bg=SURFACE, fg=MUTED,
                            relief='flat', padx=14, pady=8, cursor='hand2',
                            command=lambda k=key: self._switch(k))
            btn.pack(side='left')
            self.tab_btns[key] = btn
            frame = tk.Frame(self.content_area, bg=SURFACE)
            builder(frame)
            self.tab_frames[key] = frame
        self._switch('logo')

    def _switch(self, key):
        for k, f in self.tab_frames.items():
            f.pack_forget()
            self.tab_btns[k].configure(bg=SURFACE, fg=MUTED)
        self.tab_frames[key].pack(fill='both', expand=True)
        self.tab_btns[key].configure(bg=CARD, fg=ACCENT)

    def _logo_tab(self, f):
        tk.Label(f, text='AI LaMa menghapus objek/logo dan merekonstruksi background secara natural.',
                 font=FONT_MONO, bg=SURFACE, fg=MUTED, wraplength=620).pack(anchor='w', pady=(0,10))
        self._btn(f, '🖌️ BUKA KUAS PENGHAPUS', self.app.start_brush_select, ACCENT2).pack(anchor='w')

    def _wm_tab(self, f):
        tk.Label(f, text='Hapus watermark teks semi-transparan (putih/abu-abu) secara otomatis.',
                 font=FONT_MONO, bg=SURFACE, fg=MUTED, wraplength=620).pack(anchor='w')
        r1 = tk.Frame(f, bg=SURFACE); r1.pack(fill='x', pady=(12,4))
        tk.Label(r1, text='Threshold (brightness)', font=FONT_MONO,
                 bg=SURFACE, fg=TEXT).pack(side='left', padx=(0,8))
        self.wm_threshold = tk.IntVar(value=190)
        tk.Scale(r1, from_=100, to=255, orient='horizontal', variable=self.wm_threshold,
                 bg=SURFACE, fg=TEXT, troughcolor=BORDER, highlightthickness=0,
                 length=200, font=FONT_MONO).pack(side='left')
        r2 = tk.Frame(f, bg=SURFACE); r2.pack(fill='x', pady=(4,12))
        tk.Label(r2, text='Blend strength       ', font=FONT_MONO,
                 bg=SURFACE, fg=TEXT).pack(side='left', padx=(0,8))
        self.wm_blend = tk.DoubleVar(value=0.85)
        tk.Scale(r2, from_=0.1, to=1.0, resolution=0.05, orient='horizontal',
                 variable=self.wm_blend, bg=SURFACE, fg=TEXT, troughcolor=BORDER,
                 highlightthickness=0, length=200, font=FONT_MONO).pack(side='left')
        self._btn(f, '▶  HAPUS WATERMARK', self.app.run_watermark, ACCENT).pack(anchor='w')

    def _bg_tab(self, f):
        method = ('rembg (AI)' if REMBG_AVAILABLE
                  else ('OpenCV GrabCut' if CV2_AVAILABLE else 'PIL Fallback'))
        tk.Label(f, text=f'Hapus background → transparan (PNG).  Metode: {method}',
                 font=FONT_MONO, bg=SURFACE, fg=MUTED, wraplength=620).pack(anchor='w')
        self._btn(f, '▶  HAPUS BACKGROUND', self.app.run_bg_remove,
                  ACCENT).pack(anchor='w', pady=(10,0))

    def _btn(self, parent, text, cmd, col):
        return tk.Button(parent, text=text, command=cmd,
                         font=('Courier New', 10, 'bold'), bg=col, fg=BG,
                         activebackground=TEXT, activeforeground=BG,
                         relief='flat', padx=14, pady=7, cursor='hand2')


class StatusBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG, pady=6)
        self.lbl = tk.Label(self, text='siap.', font=FONT_MONO, bg=BG, fg=MUTED, anchor='w')
        self.lbl.pack(side='left', padx=16)
        self.progress = ttk.Progressbar(self, length=220, mode='indeterminate')
        self.progress.pack(side='right', padx=16)

    def set(self, msg, colour=None):
        self.lbl.configure(text=msg, fg=colour or MUTED)
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
        self.title('✨ AI Magic Eraser — Kuas')
        self.callback = callback

        self.display_img = img.copy()
        self.display_img.thumbnail((1000, 700), Image.LANCZOS)
        self.mask_img  = Image.new('L', self.display_img.size, 0)
        self.mask_draw = ImageDraw.Draw(self.mask_img)

        toolbar = tk.Frame(self, bg=SURFACE, pady=10, padx=10)
        toolbar.pack(fill='x')
        tk.Label(toolbar, text='Ukuran Kuas:', bg=SURFACE, fg=TEXT,
                 font=FONT_MAIN).pack(side='left')
        self.brush_size = tk.Scale(toolbar, from_=5, to=80, orient='horizontal',
                                   bg=SURFACE, fg=ACCENT, highlightthickness=0,
                                   length=200, troughcolor=BORDER)
        self.brush_size.set(20)
        self.brush_size.pack(side='left', padx=10)
        tk.Button(toolbar, text='✨ Generate AI', command=self.apply,
                  bg=ACCENT2, fg=BG, font=FONT_HEAD, cursor='hand2',
                  relief='flat', padx=10).pack(side='right', padx=5)
        tk.Button(toolbar, text='Reset Kuas', command=self.clear_brush,
                  bg=BORDER, fg=TEXT, font=FONT_MONO, cursor='hand2',
                  relief='flat', padx=10).pack(side='right', padx=5)

        tk.Label(self,
                 text='Coret area yang ingin dihapus. Pastikan tertutup sempurna.',
                 bg=BG, fg=WARNING, font=FONT_MONO, pady=5).pack()

        self.tk_img = ImageTk.PhotoImage(self.display_img)
        self.canvas = tk.Canvas(self, width=self.display_img.width,
                                height=self.display_img.height,
                                cursor='circle', bg=BG, highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
        self.last_x = self.last_y = None
        self.canvas.bind('<Button-1>',        self.start_draw)
        self.canvas.bind('<B1-Motion>',       self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

    def start_draw(self, e):
        self.last_x, self.last_y = e.x, e.y

    def draw(self, e):
        if self.last_x is not None:
            r = self.brush_size.get()
            self.canvas.create_line(self.last_x, self.last_y, e.x, e.y,
                                    fill='#FF3CAC', width=r,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND, tags='brush')
            self.mask_draw.line([self.last_x, self.last_y, e.x, e.y],
                                fill=255, width=r, joint='curve')
            self.last_x, self.last_y = e.x, e.y

    def stop_draw(self, e):
        self.last_x = self.last_y = None

    def clear_brush(self):
        self.canvas.delete('brush')
        self.mask_img  = Image.new('L', self.display_img.size, 0)
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
        self.title('Image Processor AI')
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(760, 560)
        self._src_img: Image.Image | None = None
        self._src_path = ''
        self._result_img: Image.Image | None = None
        self._build()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f'1060x640+{(sw-1060)//2}+{(sh-640)//2}')

    def _build(self):
        ttk.Style().configure('TProgressbar', troughcolor=BORDER, background=ACCENT, thickness=4)
        HeaderBar(self).pack(fill='x')
        row = tk.Frame(self, bg=BG)
        row.pack(fill='both', expand=False, padx=16, pady=(0,8))
        self.input_panel  = ImagePanel(row, self)
        self.output_panel = OutputPanel(row, self)
        self.input_panel.pack(side='left',  fill='both', expand=True, padx=(0,6))
        self.output_panel.pack(side='right', fill='both', expand=True, padx=(6,0))
        self.tools  = ToolPanel(self, self)
        self.tools.pack(fill='x', padx=16, pady=(0,8))
        self.status = StatusBar(self)
        self.status.pack(fill='x')

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp *.tiff'), ('All', '*.*')])
        if not path: return
        try:
            self._src_img  = Image.open(path).convert('RGBA')
            self._src_path = path
            self.input_panel.set_image(self._src_img, path)
            self.output_panel.clear()
            self._result_img = None
            self.status.set(f'Dibuka: {os.path.basename(path)}', SUCCESS)
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def reset_image(self):
        self._src_img = self._result_img = None
        self.input_panel.preview_lbl.configure(image='')
        self.input_panel.info_lbl.configure(text='← klik untuk upload gambar', fg=MUTED)
        self.output_panel.clear()
        self.status.set('reset.', MUTED)

    def save_image(self):
        # Cek apakah variabel memori sudah terisi
        if self._result_img is None:
            messagebox.showwarning("Peringatan", "Data gambar belum siap di memori. Silakan tunggu atau ulangi proses.")
            return

        # Paksa jendela muncul di depan (Solusi macOS)
        self.update()
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Simpan Hasil",
            defaultextension=".png",
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg'), ('All', '*.*')])

        if not path:
            return

        try:
            # Gunakan copy agar data asli tidak rusak saat dikonversi
            save_img = self._result_img.copy()
            if path.lower().endswith(('.jpg', '.jpeg')) and save_img.mode == 'RGBA':
                save_img = save_img.convert('RGB')
            
            save_img.save(path)
            messagebox.showinfo("Sukses", f"Gambar berhasil disimpan di:\n{path}")
        except Exception as e:
            messagebox.showerror("Gagal Simpan", f"Terjadi kesalahan: {e}")
    def start_brush_select(self):
        if not self._src_img:
            messagebox.showwarning('Peringatan', 'Upload gambar terlebih dahulu.')
            return
        BrushSelector(self, self._src_img.copy(), self._on_brush)

    def _on_brush(self, mask):
        self._run(lambda img: remove_logo_mask(img, mask), 'AI LaMa Inpainting')

    def _run(self, fn, label):
        if not self._src_img:
            messagebox.showwarning('Peringatan', 'Upload gambar terlebih dahulu.')
            return

        def worker():
            self.after(0, lambda: self.status.set(f'{label} berjalan... ⏳', WARNING))
            self.after(0, lambda: self.status.busy(True))
            try:
                # Proses gambar
                result = fn(self._src_img.copy())
                
                # UPDATE KRITIS: Simpan hasil ke memori thread utama
                self.after(0, lambda r=result: self._finalize_result(r, label))
                
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: self.status.set(f'Error: AI Gagal', ACCENT2))
                self.after(0, lambda: messagebox.showerror('AI Error', err_msg))
            finally:
                self.after(0, lambda: self.status.busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _finalize_result(self, result, label):
        """Fungsi pembantu untuk menyimpan hasil ke thread utama."""
        self._result_img = result # Pastikan variabel ini terisi
        self.output_panel.set_image(result, label)
        self.status.set(f'Selesai: {label} ✨', SUCCESS)

    def run_watermark(self):
        t, b = self.tools.wm_threshold.get(), self.tools.wm_blend.get()
        self._run(lambda img: remove_watermark(img, t, b), 'Hapus Watermark')

    def run_bg_remove(self):
        self._run(remove_background, 'Hapus Background')


if __name__ == '__main__':
    app = App()
    app.mainloop()