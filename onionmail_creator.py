#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import socket
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import secrets
import string

# Optional stealth plugin
try:
    # pip install playwright-stealth
    from playwright_stealth import stealth_async as stealth_plugin
except Exception:
    stealth_plugin = None


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "t", "yes", "y", "on")


def sanitize_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def parse_proxy_line(line: str) -> Optional[Dict[str, str]]:
    raw = line.strip()
    if not re.match(r"^\w+://", raw):
        raw = "http://" + raw
    try:
        u = urllib.parse.urlsplit(raw)
    except Exception:
        return None
    if not u.scheme or not u.hostname or not u.port:
        return None
    proxy = {
        "server": f"{u.scheme}://{u.hostname}:{u.port}"
    }
    if u.username:
        proxy["username"] = urllib.parse.unquote(u.username)
    if u.password:
        proxy["password"] = urllib.parse.unquote(u.password)
    return proxy


def random_letters(n: int) -> str:
    alpha = string.ascii_lowercase
    rng = secrets.SystemRandom()
    return "".join(rng.choice(alpha) for _ in range(n))


def random_digits(n: int) -> str:
    digits = string.digits
    rng = secrets.SystemRandom()
    return "".join(rng.choice(digits) for _ in range(n))


def generate_username() -> str:
    # Matches ^[A-Za-z0-9][A-Za-z\-\.0-9]*$ — e.g., fkm37592
    return random_letters(3) + random_digits(5)


def generate_strong_password(length: int = 16) -> str:
    lowers = string.ascii_lowercase
    uppers = string.ascii_uppercase
    nums = string.digits
    syms = "!@#$%^&*_-+="
    rng = secrets.SystemRandom()

    must = [
        rng.choice(lowers),
        rng.choice(uppers),
        rng.choice(nums),
        rng.choice(syms),
    ]
    allchars = lowers + uppers + nums + syms
    remaining = [rng.choice(allchars) for _ in range(max(0, length - len(must)))]
    arr = must + remaining
    rng.shuffle(arr)
    return "".join(arr)


def fetch_male_name_blocking() -> str:
    try:
        req = urllib.request.Request(
            "https://randomuser.me/api?gender=male",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            if resp.status != 200:
                return "John Smith"
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            first = data.get("results", [{}])[0].get("name", {}).get("first", "John")
            last = data.get("results", [{}])[0].get("name", {}).get("last", "Smith")
            return f"{first} {last}"
    except Exception:
        return "John Smith"


async def get_male_name() -> str:
    return await asyncio.to_thread(fetch_male_name_blocking)


@dataclass
class Args:
    mode: str
    proxies: str
    accounts: int
    concurrency: int
    headless: bool
    tor_socks: str
    output: str
    timeout_ms: int
    click_delay_ms: int
    user_agent: Optional[str]
    browser: str
    stealth: bool


def is_port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def resolve_tor_socks(tor_socks: str) -> Tuple[Optional[str], str]:
    """
    Returns (resolved, message)
    If tor_socks == 'auto', tries 127.0.0.1:9150 then 127.0.0.1:9050
    """
    if tor_socks != "auto":
        host, sep, port = tor_socks.partition(":")
        if not sep or not port.isdigit():
            return None, f"Invalid --tor-socks value: {tor_socks}. Use host:port or 'auto'."
        if is_port_open(host, int(port)):
            return tor_socks, f"Using Tor SOCKS at {tor_socks}"
        return None, f"Tor SOCKS not reachable at {tor_socks}. Is Tor Browser running? Correct port?"
    # Auto-detect
    candidates = ["127.0.0.1:9150", "127.0.0.1:9050"]
    for cand in candidates:
        h, p = cand.split(":")
        if is_port_open(h, int(p)):
            return cand, f"Auto-detected Tor SOCKS at {cand}"
    return None, "Auto-detect failed. Start Tor Browser and try --tor-socks=127.0.0.1:9150"


def pick_browser_type(p, browser: str):
    b = browser.lower()
    if b in ("firefox",):
        return p.firefox, {}
    if b in ("chrome", "msedge"):
        channel = "chrome" if b == "chrome" else "msedge"
        return p.chromium, {"channel": channel}
    return p.chromium, {}


REALISTIC_UA_CHROME_WIN = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def add_basic_stealth_to_context(context):
    # Run before any page scripts
    context.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
    window.chrome = window.chrome || { runtime: {} };
    Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
      get: function() { return window; }
    });
    const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
    if (originalQuery) {
      window.navigator.permissions.query = (parameters) => (
        parameters && parameters.name === 'notifications'
          ? Promise.resolve({ state: Notification.permission })
          : originalQuery(parameters)
      );
    }
    Object.defineProperty(window, 'devicePixelRatio', { get: () => 1 });
    try {
      Object.defineProperty(Navigator.prototype, 'platform', { get: () => 'Win32' });
    } catch(e) {}
    """)


async def create_account_attempt(
    idx: int,
    p,
    proxy_config: Optional[Dict[str, str]],
    headless: bool,
    timeout_ms: int,
    click_delay_ms: int,
    user_agent: Optional[str],
    output_file: str,
    file_lock: asyncio.Lock,
    browser_choice: str,
    stealth: bool,
) -> Dict:
    browser_type, extra_launch = pick_browser_type(p, browser_choice)

    launch_kwargs = {
        "headless": headless,
        "ignore_default_args": ["--enable-automation"],
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--no-default-browser-check",
            "--no-first-run",
            "--no-service-autorun",
            "--password-store=basic",
        ]
    }
    launch_kwargs.update(extra_launch)

    if proxy_config:
        proxy = {"server": proxy_config["server"], "bypass": "localhost,127.0.0.1"}
        if "username" in proxy_config:
            proxy["username"] = proxy_config["username"]
        if "password" in proxy_config:
            proxy["password"] = proxy_config["password"]
        launch_kwargs["proxy"] = proxy

    browser = await browser_type.launch(**launch_kwargs)

    # Incognito context (non-persistent)
    context = await browser.new_context(
        ignore_https_errors=True,
        viewport={"width": 1280, "height": 800},
        user_agent=user_agent or REALISTIC_UA_CHROME_WIN,
        locale="en-US",
    )

    # Apply basic stealth if requested
    if stealth:
        add_basic_stealth_to_context(context)

    page = await context.new_page()

    # Optional plugin-based stealth (if installed)
    if stealth and stealth_plugin is not None:
        try:
            await stealth_plugin(page)
        except Exception:
            pass

    page.set_default_timeout(timeout_ms)

    try:
        username = generate_username()
        password = generate_strong_password(16)
        full_name = await get_male_name()

        await page.goto("https://onionmail.org/account/create", wait_until="domcontentloaded", timeout=timeout_ms)

        # If there's a browser check, give it a moment
        try:
            await page.wait_for_selector("text=Checking your browser", timeout=3000)
            await page.wait_for_timeout(7000)
        except Exception:
            pass

        await page.locator("#username").fill(username)
        await page.locator("#name").fill(full_name)
        await page.locator("#password").fill(password)
        await page.locator("#password2").fill(password)

        remember = page.locator('input[name="remember"]')
        if await remember.count():
            await remember.check()

        await page.wait_for_timeout(click_delay_ms)

        # Click the "Create new account" button
        start_url = page.url
        clicked = False
        try:
            await page.locator("button.g-recaptcha.btn.btn-success").click()
            clicked = True
        except PlaywrightTimeoutError:
            pass
        except Exception:
            pass

        if not clicked:
            try:
                await page.get_by_role("button", name="Create new account").click()
                clicked = True
            except Exception:
                pass

        # Wait for URL to change after click (navigation implies likely success)
        url_changed = False
        if clicked:
            try:
                # Wait for main frame URL to become something other than start_url
                pattern = re.compile(rf"^(?!{re.escape(start_url)}$).+")
                await page.wait_for_url(pattern, wait_until="domcontentloaded", timeout=15000)
                url_changed = (page.url != start_url)
            except Exception:
                url_changed = False

        if url_changed:
            # Extra 5 seconds to be sure the account is actually created
            await page.wait_for_timeout(5000)

            # Save credentials only on URL change
            async with file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{username}:{password}\n")

            return {"success": True, "username": username, "password": password, "navigated_to": page.url}
        else:
            return {"success": False, "error": "No URL change after submit (possibly CAPTCHA or blocked)"}

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            await context.close()
        except Exception:
            pass
        try:
            await browser.close()
        except Exception:
            pass


async def main_async(a: Args):
    if a.mode not in {"direct", "proxies", "tor"}:
        print(f"Invalid --mode={a.mode}. Use direct | proxies | tor")
        sys.exit(1)

    # Resolve Tor SOCKS
    if a.mode == "tor":
        resolved, msg = resolve_tor_socks(a.tor_socks)
        print(msg)
        if not resolved:
            print("Tips:\n"
                  " - Keep Tor Browser open on Windows (it listens on 127.0.0.1:9150 by default)\n"
                  " - Or install Tor Expert Bundle and run a SOCKS port on 9050\n"
                  " - PowerShell: Test-NetConnection -ComputerName 127.0.0.1 -Port 9150")
            sys.exit(1)
        a.tor_socks = resolved

    proxies_raw = []
    if a.mode == "proxies":
        if not os.path.exists(a.proxies):
            print(f"Proxies file not found: {a.proxies}")
            sys.exit(1)
        proxies_raw = sanitize_lines(open(a.proxies, "r", encoding="utf-8").read())
        if not proxies_raw:
            print(f"No proxies found in {a.proxies}")
            sys.exit(1)

    print(f"Mode: {a.mode}")
    if a.mode == "proxies":
        print(f"Loaded {len(proxies_raw)} proxies from {a.proxies}")
    if a.mode == "tor":
        print(f"Using Tor SOCKS5 at {a.tor_socks}")
    print(f"Accounts to attempt: {a.accounts}, Concurrency: {a.concurrency}, Headless: {a.headless}, Browser: {a.browser}, Stealth: {a.stealth}")

    file_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, a.concurrency))

    async with async_playwright() as p:
        tasks = []
        for i in range(a.accounts):
            proxy_cfg = None
            if a.mode == "tor":
                proxy_cfg = {"server": f"socks5://{a.tor_socks}"}
            elif a.mode == "proxies":
                line = proxies_raw[i % len(proxies_raw)]
                parsed = parse_proxy_line(line)
                if parsed is None:
                    print(f"[{i+1}/{a.accounts}] Skipping invalid proxy line: {line}")
                else:
                    proxy_cfg = parsed

            async def runner(index: int, cfg: Optional[Dict[str, str]]):
                async with sem:
                    via = f" via {cfg['server']}" if cfg else ""
                    print(f"[{index}/{a.accounts}] Starting attempt{via}...")
                    res = await create_account_attempt(
                        idx=index,
                        p=p,
                        proxy_config=cfg,
                        headless=a.headless,
                        timeout_ms=a.timeout_ms,
                        click_delay_ms=a.click_delay_ms,
                        user_agent=a.user_agent,
                        output_file=a.output,
                        file_lock=file_lock,
                        browser_choice=a.browser,
                        stealth=a.stealth,
                    )
                    if res.get("success"):
                        print(f"[{index}/{a.accounts}] Saved: {res['username']}:******** (url: {res.get('navigated_to')})")
                    else:
                        print(f"[{index}/{a.accounts}] Failed: {res.get('error', 'Unknown error')}")
                    return res

            tasks.append(asyncio.create_task(runner(i + 1, proxy_cfg)))

        results = await asyncio.gather(*tasks, return_exceptions=False)
        success_count = sum(1 for r in results if r.get("success"))
        print(f"Done. Success attempts: {success_count}/{a.accounts}")
        print(f"Credentials appended to {a.output} only when URL changed and after a 5s confirmation wait.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OnionMail account creator (Playwright, Python)")
    p.add_argument("--mode", default="direct", choices=["direct", "proxies", "tor"], help="Connection mode")
    p.add_argument("--proxies", default="proxies.txt", help="Path to proxies file when --mode=proxies")
    p.add_argument("--accounts", type=int, default=1, help="Number of account attempts")
    p.add_argument("--concurrency", type=int, default=1, help="Concurrent attempts")
    p.add_argument("--headless", type=str, default="true", help="true|false")
    p.add_argument("--tor-socks", dest="tor_socks", default="auto", help="Tor SOCKS address or 'auto' (Windows Tor Browser usually 127.0.0.1:9150)")
    p.add_argument("--output", default="onionmail_accounts.txt", help="Output credentials file")
    p.add_argument("--timeout", dest="timeout_ms", type=int, default=60000, help="Per-page operation timeout (ms)")
    p.add_argument("--click-delay", dest="click_delay_ms", type=int, default=200, help="Delay before clicking (ms)")
    p.add_argument("--user-agent", dest="user_agent", default=None, help="Optional custom User-Agent")
    p.add_argument("--browser", dest="browser", default="chromium", choices=["chromium", "chrome", "msedge", "firefox"], help="Browser engine/channel")
    p.add_argument("--stealth", dest="stealth", default="true", help="Enable stealth JS tweaks (true|false)")
    return p


def parse_args() -> Args:
    p = build_arg_parser()
    ns = p.parse_args()
    return Args(
        mode=ns.mode.lower(),
        proxies=ns.proxies,
        accounts=max(1, int(ns.accounts)),
        concurrency=max(1, int(ns.concurrency)),
        headless=str2bool(ns.headless),
        tor_socks=str(ns.tor_socks),
        output=ns.output,
        timeout_ms=int(ns.timeout_ms),
        click_delay_ms=int(ns.click_delay_ms),
        user_agent=ns.user_agent,
        browser=ns.browser,
        stealth=str2bool(ns.stealth),
    )


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user.")