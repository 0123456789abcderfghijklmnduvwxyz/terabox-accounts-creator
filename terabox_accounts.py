import argparse
import os
import re
import sys
import time
import math
import threading
import secrets
import string
import socket
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from urllib.parse import urlparse
from contextlib import suppress
import base64
from io import BytesIO
from PIL import Image
import easyocr

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError, Error as PWError

# Optional stealth
try:
    from playwright_stealth import stealth_sync
except ImportError:
    stealth_sync = None

ACCOUNTS_FILE = "onionmail_accounts.txt"
OUTPUT_FILE = "terabox_accounts.txt"
PROXIES_FILE = "proxies.txt"

accounts_lock = threading.Lock()
output_lock = threading.Lock()
print_lock = threading.Lock()  # keep log lines tidy
reader = easyocr.Reader(['en'], gpu=False)

# ====================== Solve Captcha's cause for some reason they are too lazy to do it correct and you can just read the code from a variable, conviniently called "code", if it fails it goes the hard way by just reading it, which will take a few attempts. ========================================== #

import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import easyocr
import re
import time
from contextlib import suppress

# Reader global initialisieren (nur einmal!)
reader = easyocr.Reader(['en'], gpu=False)

def _bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert("RGBA")

def _pil_preprocess(img: Image.Image, scale: int = 2, do_threshold: bool = True) -> Image.Image:
    """Vorverarbeitung: Graustufen, Resize, Autocontrast, Threshold, Glättung"""
    g = img.convert("L")
    w, h = g.size
    g = g.resize((max(32, w * scale), max(32, h * scale)), resample=Image.BILINEAR)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    if do_threshold:
        hist = g.histogram()
        pixels = sum(i * hist[i] for i in range(256))
        total = sum(hist) or 1
        mean = pixels // total
        thresh = int(max(100, min(200, mean)))
        g = g.point(lambda p: 255 if p > thresh else 0)
    return g

def _easyocr_read(img_pil: Image.Image):
    """
    OCR mit EasyOCR, optimiert für 4 Zeichen [A-Z0-9]
    """
    # PIL → numpy
    img_np = np.array(img_pil)

    # EasyOCR erkennt Textbereiche
    results = reader.readtext(img_np, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    if not results:
        return None

    # Ergebnisse kombinieren, filtern
    text = "".join(results).strip().upper()
    if re.fullmatch(r"[A-Z0-9]{4}", text):
        return text
    return None
    
def extract_captcha_image_bytes(page, selector_candidates=None) -> bytes | None:
    sel_cands = selector_candidates or ["canvas", "#canvas", "#captcha-img", "img.captcha", "img"]

    # 1) Canvas → toDataURL
    try:
        handle = page.query_selector("canvas")
        if handle:
            img_base64 = page.evaluate(
                """
                (cnv) => {
                    var tmp = document.createElement('canvas');
                    tmp.width = cnv.width;
                    tmp.height = cnv.height;
                    tmp.getContext('2d').drawImage(cnv, 0, 0);
                    return tmp.toDataURL('image/png').substring(22);
                }
                """,
                handle
            )
            if img_base64:
                return base64.b64decode(img_base64)
    except Exception:
        pass

    # 2) img[src^=data]
    for sel in sel_cands:
        try:
            src = page.evaluate("sel => { const el = document.querySelector(sel); return el ? el.src : null; }", sel)
            if src and src.startswith("data:"):
                header, b64 = src.split(",", 1)
                return base64.b64decode(b64)
        except Exception:
            pass

    # 3) fetch(src)
    for sel in sel_cands:
        try:
            src = page.evaluate("sel => { const el = document.querySelector(sel); return el ? el.src : null; }", sel)
            if src and not src.startswith("data:"):
                b64 = page.evaluate(
                    """(url) => fetch(url).then(r => r.arrayBuffer()).then(b => {
                        const u8 = new Uint8Array(b);
                        let bin = '';
                        for (let i=0;i<u8.length;i++){ bin += String.fromCharCode(u8[i]); }
                        return btoa(bin);
                    }).catch(e => null)""",
                    src
                )
                if b64:
                    return base64.b64decode(b64)
        except Exception:
            pass

    # 4) Screenshot des Elements
    for sel in sel_cands:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                return loc.first.screenshot()
        except Exception:
            pass

    # 5) Screenshot Ausschnitt
    for sel in sel_cands:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                box = loc.first.bounding_box()
                if box:
                    return page.screenshot(clip={"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]})
        except Exception:
            pass

    return None

def fill_captcha_and_confirm(page, guess: str) -> bool:
    """inputs captcha and presses the confirm button"""
    filled = False

    try:
        page.fill("#input", guess)
        filled = True
    except Exception as e:
        log(f"Konnte Input #input nicht füllen: {e}")
        return False

    clicked = False
    try:
        page.get_by_role("button", name="Confirm").click(timeout=3000)
        clicked = True
    except Exception:
        btn_selectors = ['button:has-text("Confirm")', 'button:has-text("OK")', 'button:has-text("Verify")']
        for bsel in btn_selectors:
            try:
                if click_if_visible(page.locator(bsel), timeout=2000, force=True):
                    clicked = True
                    break
            except Exception:
                pass

    return filled and clicked

def solve_simple_verify_captcha(page, worker_slot: int, acct_idx: int = None, max_attempts: int = 20) -> bool:
    """Captcha-Solver using ONLY direct access to JS variable `code`, with retries"""
    log("Got a Captcha (direct variable read).", acct_idx=acct_idx, worker_slot=worker_slot)

    for attempt in range(1, max_attempts + 1):
        try:
            # Read captcha directly from JS variable
            code = page.evaluate("() => window.code")
            if not code:
                log(f"[Attempt {attempt}] Could not read captcha variable.", acct_idx=acct_idx, worker_slot=worker_slot)
                continue

            log(f"[Attempt {attempt}] JS captcha code: '{code}'", acct_idx=acct_idx, worker_slot=worker_slot)

            # Fill the captcha input and confirm
            fill_captcha_and_confirm(page, code)
            time.sleep(1.5)

            # Verify captcha solved
            if not page.url.startswith("https://www.terabox.com/simple-verify"):
                log("Captcha solved via JS variable (URL changed).", acct_idx=acct_idx, worker_slot=worker_slot)
                return True

            still_here = page.locator("#canvas").count() > 0
            if not still_here:
                log("Captcha disappeared, solved.", acct_idx=acct_idx, worker_slot=worker_slot)
                return True

            log(f"[Attempt {attempt}] Captcha not accepted, retrying...", acct_idx=acct_idx, worker_slot=worker_slot)

        except Exception as e:
            log(f"[Attempt {attempt}] Error while reading captcha variable: {e}", acct_idx=acct_idx, worker_slot=worker_slot)

    log(f"Failed to solve captcha after {max_attempts} attempts.", acct_idx=acct_idx, worker_slot=worker_slot)
    return False


# ------------------------------- Logging ---------------------------------- #

def log(msg: str, acct_idx: Optional[int] = None, worker_slot: Optional[int] = None):
    """
    Unified log function: logs message with optional account index and worker slot.
    Examples:
        log("Hello")                             -> [MAIN] Hello
        log("Hello", worker_slot=1)              -> [W01] Hello
        log("Hello", acct_idx=3)                 -> [#003] Hello
        log("Hello", acct_idx=3, worker_slot=1)  -> [#003 W01] Hello
    """
    with print_lock:
        if acct_idx is not None and worker_slot is not None:
            prefix = f"[#{acct_idx:03d} W{worker_slot:02d}]"
        elif acct_idx is not None:
            prefix = f"[#{acct_idx:03d}]"
        elif worker_slot is not None:
            prefix = f"[W{worker_slot:02d}]"
        else:
            prefix = "[MAIN]"
        print(f"{prefix} {msg}", flush=True)

# ----------------------------- File helpers ------------------------------- #

def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())

account_counter = 0
account_counter_lock = threading.Lock()

def pop_onion_account(path: str) -> Tuple[str, str, int]:
    """
    Holt den nächsten unbenutzten username:password aus onionmail_accounts.txt.
    Prüft gegen used_terabox_emails.txt und protokolliert dort die genutzten Accounts.
    """
    global account_counter
    with accounts_lock:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")

        # Alle möglichen Accounts laden
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise RuntimeError("No accounts left in onionmail_accounts.txt")

        # Bereits benutzte Mails laden
        used = set()
        if os.path.exists("used_terabox_emails.txt"):
            with open("used_terabox_emails.txt", "r", encoding="utf-8") as f:
                used = {ln.strip().lower() for ln in f if ln.strip()}

        # Nächsten unbenutzten Account finden
        username, pwd = None, None
        for entry in lines:
            parts = entry.split(":", 1)
            if len(parts) != 2:
                continue
            u, p = parts[0].strip(), parts[1].strip()
            email = f"{u}@onionmail.org".lower()
            if email not in used:
                username, pwd = u, p
                # sofort als benutzt markieren
                with open("used_terabox_emails.txt", "a", encoding="utf-8") as f:
                    f.write(email + "\n")
                break

        if not username:
            raise RuntimeError("No unused accounts left in onionmail_accounts.txt")

    with account_counter_lock:
        account_counter += 1
        idx = account_counter

    return username, pwd, idx

def save_terabox_credentials(email: str, password: str):
    with output_lock:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"{email}:{password}\n")


# ------------------------------- Password --------------------------------- #

def generate_password(min_len: int = 12) -> str:
    """
    Generate a secure password of at least min_len characters,
    containing at least one uppercase letter and one digit.
    """
    alphabet = string.ascii_letters + string.digits + "_-@!#$%?"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(max(min_len, 12)))
        if any(c.isupper() for c in pw) and any(c.isdigit() for c in pw):
            return pw


# ------------------------------- Proxies ---------------------------------- #

def parse_proxy_line(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "://" not in line:
        # default to http if scheme omitted
        line = "http://" + line
    u = urlparse(line)
    if not u.scheme or not u.hostname or not u.port:
        return None
    server = f"{u.scheme}://{u.hostname}:{u.port}"
    cfg = {"server": server}
    if u.username:
        cfg["username"] = u.username
    if u.password:
        cfg["password"] = u.password
    return cfg


def load_proxies(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found for proxy mode")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            cfg = parse_proxy_line(ln)
            if cfg:
                out.append(cfg)
    if not out:
        raise RuntimeError("No valid proxies parsed from proxies.txt")
    return out


# -------------------------------- Tor ------------------------------------- #

def detect_tor_socks() -> Tuple[str, int]:
    # Prefer common ports based on OS; verify by attempting to connect.
    host = "127.0.0.1"
    if sys.platform.startswith("win"):
        candidates = [9150, 9050, 9151, 9051]
    else:
        candidates = [9050, 9150, 9051, 9151]

    for port in candidates:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return host, port
        except Exception:
            continue

    raise RuntimeError("Could not auto-detect Tor SOCKS port on 127.0.0.1. "
                       "Start Tor Browser (9150) or Tor service (9050), or set --tor-socks=host:port")


def parse_tor_socks(value: str) -> Tuple[str, int]:
    if value.lower() == "auto":
        return detect_tor_socks()
    # Accept "port" or "host:port"
    if ":" in value:
        host, port = value.split(":", 1)
        return host.strip() or "127.0.0.1", int(port)
    else:
        return "127.0.0.1", int(value)


# ------------------------------- Playwright -------------------------------- #

def build_proxy(mode: str, worker_slot: int, proxies: Optional[List[Dict]], tor_host: Optional[str], tor_port: Optional[int]) -> Optional[Dict]:
    if mode == "direct":
        return None
    if mode == "proxy":
        if not proxies:
            raise ValueError("Proxy mode requires proxies loaded from proxies.txt")
        idx = (worker_slot - 1) % len(proxies)
        return proxies[idx]
    if mode == "tor":
        if not tor_host or not tor_port:
            raise ValueError("Tor mode requires tor_host and tor_port")
        return {"server": f"socks5://{tor_host}:{tor_port}"}
    raise ValueError(f"Unknown mode: {mode}")


def user_agent_for(browser_name: str, os_name: str = "") -> str:
    if browser_name.lower() in ("chrome", "chromium", "edge", "msedge"):
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    if browser_name.lower() == "firefox":
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0"
    if browser_name.lower() in ("webkit", "safari"):
        return "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version=17.5 Safari/605.1.15"
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


def apply_stealth_if_requested(page, stealth: bool, worker_slot: int):
    if stealth:
        if stealth_sync is None:
            log("Stealth requested but playwright-stealth not installed. Run: pip install playwright-stealth", worker_slot=worker_slot)
        else:
            try:
                stealth_sync(page)
            except Exception as e:
                log(f"Stealth application failed (continuing): {e}", worker_slot=worker_slot)


# ---------------------------- Window management --------------------------- #

def get_screen_size() -> Tuple[int, int]:
    # Try Tkinter; fallback to 1920x1080
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def compute_window_grid(n: int) -> List[Tuple[int, int, int, int]]:
    # Returns a list of (x, y, w, h) for up to n windows
    screen_w, screen_h = get_screen_size()
    cols = max(1, math.ceil(math.sqrt(max(n, 1))))
    rows = max(1, math.ceil(n / max(cols, 1)))
    margin = 40
    cell_w = max(820, screen_w // cols - margin)
    cell_h = max(520, screen_h // rows - margin)
    positions = []
    for i in range(n):
        r = i // cols
        c = i % cols
        x = c * (screen_w // cols) + margin // 2
        y = r * (screen_h // rows) + margin // 2
        positions.append((x, y, cell_w, cell_h))
    return positions


# ------------------------------ Page helpers ------------------------------ #

def click_if_visible(locator, timeout: int = 2000, force: bool = False) -> bool:
    try:
        locator.first.wait_for(state="visible", timeout=timeout)
        locator.first.click(force=force)
        return True
    except Exception:
        return False


def wait_for_text(page, text: str, timeout: int = 30000) -> bool:
    try:
        page.locator(f"text={text}").wait_for(state="visible", timeout=timeout)
        return True
    except PWTimeoutError:
        return False


def safe_reload(page, worker_slot: int):
    try:
        page.reload(wait_until="domcontentloaded", timeout=15000)
        return True
    except Exception:
        # Fallback to keyboard refresh
        try:
            page.bring_to_front()
            page.keyboard.press("F5")
            time.sleep(1.5)
            return True
        except Exception as e:
            log(f"Reload failed: {e}", worker_slot=worker_slot)
            return False


def extract_code_from_text(text: str) -> Optional[str]:
    # Look for: "XXXX is your TeraBox verification code"
    m = re.search(r"(\d{4})\s+is your TeraBox verification code", text or "", flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def interleaved_wait_and_collect_code(page_tera, page_onion, worker_slot: int,
                                      max_wait_prompt: int = 120, max_wait_code: int = 180) -> Tuple[bool, Optional[str]]:
    """
    Interleaves checking the TeraBox prompt and polling OnionMail for the code.
    Returns (tera_prompt_visible, code or None)
    """
    tera_prompt_seen = False
    code_found = None
    last_onion_refresh = 0.0
    start = time.time()
    prompt_deadline = start + max_wait_prompt
    code_deadline = start + max_wait_code

    log("Starting interleaved wait: TeraBox prompt + OnionMail refresh", worker_slot=worker_slot)

    while True:
        now = time.time()

        # Check TeraBox prompt or the code input fields existence
        if not tera_prompt_seen:
            try:
                if page_tera.locator('text=Please enter verification code').count() > 0:
                    page_tera.locator('text=Please enter verification code').first.wait_for(state="visible", timeout=1000)
                    tera_prompt_seen = True
                else:
                    # Alternative: inputs appeared
                    inputs = page_tera.locator("input.new-card-verify-input")
                    if inputs.count() > 0:
                        inputs.first.wait_for(state="visible", timeout=1000)
                        tera_prompt_seen = True
            except Exception:
                pass

        # Refresh OnionMail every ~5 seconds
        if now - last_onion_refresh > 5 and now < code_deadline and code_found is None:
            log("Refreshing OnionMail to check for verification email...", worker_slot=worker_slot)
            page_onion.bring_to_front()
            safe_reload(page_onion, worker_slot)
            with suppress(Exception):
                # Quick subject line check
                possible = page_onion.locator("text=/\\d{4}\\s+is your TeraBox verification code/i")
                if possible.count() > 0:
                    txt = possible.first.inner_text(timeout=2000)
                    code = extract_code_from_text(txt or "")
                    if code:
                        code_found = code

            if code_found is None:
                with suppress(Exception):
                    body_text = page_onion.inner_text("body", timeout=4000)
                    code = extract_code_from_text(body_text or "")
                    if code:
                        code_found = code

            last_onion_refresh = now

        # Break conditions
        if tera_prompt_seen and code_found:
            return True, code_found

        if now > prompt_deadline and not tera_prompt_seen and code_found:
            # We do have code, but prompt hasn't shown — return what we have
            return False, code_found

        if now > code_deadline and code_found is None and tera_prompt_seen:
            return tera_prompt_seen, None

        if now > prompt_deadline and now > code_deadline:
            return tera_prompt_seen, code_found

        time.sleep(1.0)


def fill_verification_code(page_tera, code: str):
    fields = page_tera.locator("input.new-card-verify-input")
    try:
        fields.first.wait_for(state="visible", timeout=20000)
        count = fields.count()
        if count >= len(code):
            for i, ch in enumerate(code):
                fields.nth(i).fill(ch)
        else:
            single = page_tera.locator('input[name*="code" i], input[placeholder*="code" i], input[type="tel"]')
            single.first.fill(code)
    except Exception:
        single = page_tera.locator('input[name*="code" i], input[placeholder*="code" i], input[type="tel"]')
        single.first.fill(code)


def wait_for_url_change(page, old_url: str, timeout: int = 90000) -> bool:
    try:
        page.wait_for_function("url => window.location.href !== url", arg=old_url, timeout=timeout)
        return True
    except Exception:
        return False


# ----------------------------- Config dataclass --------------------------- #

@dataclass
class Config:
    mode: str
    headless: bool
    stealth: bool
    browser_name: str
    threads: int
    accounts: int
    tor_host: Optional[str]
    tor_port: Optional[int]
    proxies: Optional[List[Dict]]


# -------------------------------- Worker ---------------------------------- #

def run_worker_slot(worker_slot: int, cfg: Config, positions: List[Tuple[int, int, int, int]], jobs_q: "queue.Queue[int]"):
    import queue  # local import for type
    # Start Playwright once per worker to avoid racey start/stop across threads
    cm = sync_playwright()
    p = cm.start()
    try:
        while True:
            try:
                _ = jobs_q.get_nowait()
            except queue.Empty:
                return

            try:
                create_one_account(worker_slot, cfg, positions[worker_slot - 1], p)
            except Exception as e:
                log(f"ERROR: {e}", worker_slot=worker_slot)
            finally:
                jobs_q.task_done()
    finally:
        with suppress(Exception):
            p.stop()


def create_one_account(worker_slot: int, cfg: Config, window_rect: Tuple[int, int, int, int], p):
    username, onion_pwd, acct_idx = pop_onion_account(ACCOUNTS_FILE)
    email = f"{username}@onionmail.org"
    log(f"Using account: {email}", acct_idx=acct_idx, worker_slot=worker_slot)

    proxy_cfg = build_proxy(cfg.mode, worker_slot, cfg.proxies, cfg.tor_host, cfg.tor_port)

    # Select browser type and channel
    browser_type = cfg.browser_name.lower()
    x, y, w, h = window_rect

    launch_kwargs = {
        "headless": cfg.headless,
        "proxy": proxy_cfg,
    }

    chromium_args = []
    if not cfg.headless:
        chromium_args.extend([f"--window-position={x},{y}", f"--window-size={w},{h}"])

    ua = user_agent_for(cfg.browser_name)

    browser = None
    context = None
    page_tera = None
    page_onion = None

    try:
        if browser_type in ("chrome", "edge", "msedge", "chromium"):
            channel = None
            bt = p.chromium
            if browser_type == "chrome":
                channel = "chrome"
            elif browser_type in ("edge", "msedge"):
                channel = "msedge"
            if channel:
                try:
                    browser = bt.launch(channel=channel, args=chromium_args, **launch_kwargs)
                except Exception as e:
                    log(f"{channel} channel not found or failed ({e}); falling back to bundled Chromium", acct_idx=acct_idx, worker_slot=worker_slot)
                    browser = bt.launch(args=chromium_args, **launch_kwargs)
            else:
                browser = bt.launch(args=chromium_args, **launch_kwargs)
        elif browser_type == "firefox":
            browser = p.firefox.launch(**launch_kwargs)
        elif browser_type in ("webkit", "safari"):
            browser = p.webkit.launch(**launch_kwargs)
        else:
            log(f"Unknown browser '{cfg.browser_name}', falling back to Chromium", acct_idx=acct_idx, worker_slot=worker_slot)
            browser = p.chromium.launch(args=chromium_args, **launch_kwargs)

        # Create context sized like the window
        context = browser.new_context(
            ignore_https_errors=True,
            viewport={"width": w - 20, "height": h - 80} if not cfg.headless else {"width": 1280, "height": 800},
            user_agent=ua,
            locale="en-US",
        )
        context.set_default_timeout(45000)

        # Block unnecessary resources to save bandwidth
        def block_unneeded(route, request):
            url = request.url.lower()
            rtype = request.resource_type

            # Block resource types that are just visual/heavy
            if rtype in ["image", "font", "media", "other", "texttrack", "imageset"]:
                return route.abort()

            # Block known 3rd party scripts we don’t need
            if any(bad in url for bad in [
                "facebook", "kakao", "apple", "google-analytics", "doubleclick",
                "googletag", "ads", "abclite", "adzerk", "analytics", "cdn.api.twitter",
                "exelator", "fontawesome", "google", "googletagmanager",
            ]):
                return route.abort()

            # Allow only terabox.com and onionmail.org + required cdn scripts
            if not any(domain in url for domain in [
                "terabox.com", "onionmail.org", "teraboxcdn.com"
            ]):
                return route.abort()

            # Everything else continues
            return route.continue_()


        context.route("**/*", block_unneeded)

        # Monitoring
        def on_page_close(pg):
            log("Page closed", acct_idx=acct_idx, worker_slot=worker_slot)
        def on_page_crash(pg):
            log("Page crashed", acct_idx=acct_idx, worker_slot=worker_slot)

        # TeraBox tab
        page_tera = context.new_page()
        page_tera.on("close", on_page_close)
        with suppress(Exception):
            page_tera.on("crash", on_page_crash)
        apply_stealth_if_requested(page_tera, cfg.stealth, worker_slot)
        if not cfg.headless:
            page_tera.bring_to_front()

        log("Opening TeraBox (recorder sequence) ...", acct_idx=acct_idx, worker_slot=worker_slot)
        page_tera.goto("https://www.terabox.com/", wait_until="domcontentloaded", timeout=60000)

        # If we got redirected to simple-verify early, handle captcha
        if page_tera.url.startswith("https://www.terabox.com/simple-verify"):
            log("Landed on simple-verify captcha page; attempting to solve...", acct_idx=acct_idx, worker_slot=worker_slot)
            solved = solve_simple_verify_captcha(page_tera, worker_slot=worker_slot, acct_idx=acct_idx)
            if not solved:
                raise RuntimeError("Could not solve captcha on simple-verify page after multiple attempts.")

        # Use your recorded commands exactly
        with suppress(Exception):
            page_tera.get_by_role("button", name="Get Started").first.click()
        with suppress(Exception):
            page_tera.locator("span").filter(has_text="Sign up").click()
        with suppress(Exception):
            page_tera.locator(".other-item > div:nth-child(2)").click()
        with suppress(Exception):
            page_tera.get_by_role("textbox", name="Enter your email").click()
        with suppress(Exception):
            page_tera.get_by_role("textbox", name="Enter your email").fill(email)
        with suppress(Exception):
            page_tera.get_by_text("Continue").click()

        # OnionMail tab
        page_onion = context.new_page()
        page_onion.on("close", on_page_close)
        with suppress(Exception):
            page_onion.on("crash", on_page_crash)
        apply_stealth_if_requested(page_onion, cfg.stealth, worker_slot)
        if not cfg.headless:
            page_onion.bring_to_front()

        log("Opening OnionMail login...", acct_idx=acct_idx, worker_slot=worker_slot)
        page_onion.goto("https://onionmail.org/account/login", wait_until="domcontentloaded", timeout=60000)

        page_onion.locator("#username").wait_for(state="visible", timeout=30000)
        page_onion.locator("#username").fill(username)
        page_onion.locator("#password").fill(onion_pwd)
        page_onion.locator(".btn-success").first.click()

        # Interleaved: wait for TeraBox prompt + poll OnionMail for code
        if not cfg.headless:
            page_tera.bring_to_front()
        log("Waiting for TeraBox prompt and polling OnionMail for verification email...", acct_idx=acct_idx, worker_slot=worker_slot)
        tera_prompt, code = interleaved_wait_and_collect_code(page_tera, page_onion, worker_slot,
                                                              max_wait_prompt=120, max_wait_code=180)

        if not tera_prompt:
            log("TeraBox prompt not detected yet; trying to proceed if inputs are present", acct_idx=acct_idx, worker_slot=worker_slot)
        if not code:
            raise TimeoutError("Verification email not found on onionmail.org within timeout")

        if not cfg.headless:
            page_tera.bring_to_front()
        log(f"Entering verification code (per-box): {code}", acct_idx=acct_idx, worker_slot=worker_slot)

        try:
            page_tera.get_by_role("textbox").first.wait_for(state="visible", timeout=5000)
        except Exception:
            pass

        if len(code) >= 4:
            with suppress(Exception): page_tera.get_by_role("textbox").first.fill(code[0])
            with suppress(Exception): page_tera.get_by_role("textbox").nth(1).fill(code[1])
            with suppress(Exception): page_tera.get_by_role("textbox").nth(2).fill(code[2])
            with suppress(Exception): page_tera.get_by_role("textbox").nth(3).fill(code[3])
        else:
            fill_verification_code(page_tera, code)

        with suppress(Exception):
            page_tera.locator(".i-look").click()

        password = generate_password(min_len=12)
        log(f"Generated password: {password}", acct_idx=acct_idx, worker_slot=worker_slot)

        filled_pw = False
        with suppress(Exception):
            page_tera.get_by_role("textbox", name="Enter your password").click()
            page_tera.get_by_role("textbox", name="Enter your password").fill(password)
            filled_pw = True

        if not filled_pw:
            pw_input = page_tera.locator('input[placeholder="Enter your password"]')
            try:
                pw_input.wait_for(state="visible", timeout=60000)
                pw_input.fill(password)
            except Exception:
                with suppress(Exception):
                    page_tera.locator('input[placeholder*="password"]').first.fill(password)

        submitted = False
        with suppress(Exception):
            page_tera.get_by_text("Create account").click()
            submitted = True

        if not submitted:
            submit_candidates = [
                'button:has-text("Sign up")',
                'button:has-text("Create")',
                'button:has-text("Continue")',
                'button:has-text("Next")',
                '[role="button"]:has-text("Sign up")',
            ]
            for sel in submit_candidates:
                if click_if_visible(page_tera.locator(sel), timeout=1500, force=True):
                    submitted = True
                    break
            if not submitted:
                with suppress(Exception):
                    try:
                        page_tera.keyboard.press("Enter")
                    except Exception:
                        pass

        old_url = page_tera.url
        log("Waiting for URL change after submission...", acct_idx=acct_idx, worker_slot=worker_slot)
        changed = wait_for_url_change(page_tera, old_url, timeout=120000)
        if not changed:
            log("URL did not change within timeout; proceeding to close anyway.", acct_idx=acct_idx, worker_slot=worker_slot)
        else:
            save_terabox_credentials(email, password)

        log("Finalizing; waiting 5 seconds...", acct_idx=acct_idx, worker_slot=worker_slot)
        time.sleep(5)

    finally:
        with suppress(Exception):
            if context:
                context.close()
        with suppress(Exception):
            if browser:
                browser.close()

    log(f"Completed task for: {email}", acct_idx=acct_idx, worker_slot=worker_slot)

# --------------------------------- Main ----------------------------------- #

def str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y", "on")


def main():
    import queue

    parser = argparse.ArgumentParser(description="TeraBox sign-up automation with Playwright")
    parser.add_argument("--mode", choices=["direct", "proxy", "tor"], default="direct",
                        help="Network mode: direct, proxy (from proxies.txt), or tor")
    parser.add_argument("--tor-socks", default="auto",
                        help="Tor SOCKS endpoint (host:port), or 'auto' to auto-detect (default: auto)")
    parser.add_argument("--browser", default="chrome",
                        help="Browser family: chrome, chromium, firefox, webkit, edge")
    parser.add_argument("--headless", default="true", help="true/false: run headless")
    parser.add_argument("--stealth", default="false", help="true/false: apply playwright-stealth")
    parser.add_argument("--accounts", type=int, default=1, help="Number of accounts to create")
    parser.add_argument("--threads", type=int, default=1, help="Maximum concurrent workers")
    # Backward-compat alias
    parser.add_argument("--concurrency", type=int, help="Alias for --threads")

    args = parser.parse_args()

    # Resolve booleans and concurrency
    headless = str2bool(args.headless)
    stealth = str2bool(args.stealth)
    threads = args.threads if args.concurrency is None else args.concurrency

    # Validate/prepare accounts
    available = count_lines(ACCOUNTS_FILE)
    requested = max(0, int(args.accounts))
    if available <= 0:
        log(f"No accounts available in {ACCOUNTS_FILE}. Nothing to do.")
        sys.exit(1)

    if requested <= 0:
        log("--accounts must be >= 1")
        sys.exit(1)

    total_to_create = min(requested, available)
    if available < requested:
        log(f"Requested {requested} accounts but only {available} available. Will create {total_to_create}.")

    # Prepare network settings
    proxies = None
    tor_host = None
    tor_port = None

    if args.mode == "proxy":
        proxies = load_proxies(PROXIES_FILE)
        log(f"Loaded {len(proxies)} proxies from {PROXIES_FILE}. Using round-robin across workers.")
    elif args.mode == "tor":
        tor_host, tor_port = parse_tor_socks(args.tor_socks)
        log(f"Tor SOCKS set to {tor_host}:{tor_port}")

    # Final config
    cfg = Config(
        mode=args.mode,
        headless=headless,
        stealth=stealth,
        browser_name=args.browser,
        threads=max(1, int(threads)),
        accounts=total_to_create,
        tor_host=tor_host,
        tor_port=tor_port,
        proxies=proxies
    )

    # Prepare worker positions for window tiling (only used when not headless)
    slots = min(cfg.threads, cfg.accounts)
    positions = compute_window_grid(slots) if not cfg.headless else [(0, 0, 1280, 800)] * slots

    # Jobs queue: exactly total_to_create jobs
    jobs_q = queue.Queue()
    for i in range(cfg.accounts):
        jobs_q.put(i)

    # Start worker slots (fixed number = slots), each consumes from queue
    threads_list = []
    for slot in range(1, slots + 1):
        t = threading.Thread(target=run_worker_slot, args=(slot, cfg, positions, jobs_q), daemon=False)
        t.start()
        threads_list.append(t)

    # Wait for completion
    for t in threads_list:
        t.join()

    log("All done.")


if __name__ == "__main__":
    main()
