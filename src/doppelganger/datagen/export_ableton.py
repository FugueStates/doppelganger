"""
Drive Ableton's "Export Audio/Video" dialog from Python (Windows only, via pywinauto).

This is the one step no API can do — Export is GUI-only. We send the export
shortcut, click Export, fill the Save dialog with our batch folder + prefix, then
wait for the rendered files to appear.

GUI automation is machine-specific, so START with the diagnostic:

    uv run python -m doppelganger.datagen.export_ableton inspect

Open Live's Export dialog (Ctrl+Shift+R) manually first, then run `inspect` — it
dumps the dialog's control identifiers so we can wire `export_batch` precisely.

Then a real export:

    uv run python -m doppelganger.datagen.export_ableton export "dataset/pending/b0000003"

Prereq: `uv sync --extra datagen`. Set Live's Export settings once (Rendered Track =
All Individual Tracks, sample rate, length); Live remembers them between exports.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import win32clipboard
    import win32gui
    from pywinauto import Application, Desktop
    from pywinauto.keyboard import send_keys
except ImportError:  # pragma: no cover
    print("pywinauto not installed. Run: uv sync --extra datagen", file=sys.stderr)
    raise


def _set_clipboard(text: str) -> None:
    """Put text on the Windows clipboard (robust path entry — no key-escaping issues)."""
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()

LIVE_TITLE_RE = r".*Ableton Live.*"
EXPORT_TITLE_RE = r".*[Ee]xport.*"
SAVE_DIALOG_CLASS = "#32770"  # standard Windows common dialog


def _desktop() -> "Desktop":
    return Desktop(backend="uia")


def find_live():
    """Return the top-level Ableton Live window (raises if not found).

    Looks the window up with raw win32 EnumWindows (instant), NOT a UIA title_re scan:
    UIA inspects every top-level window out-of-process, and with hundreds of windows
    open that scan can take MINUTES — the export then looks "hung" before a single
    keystroke is ever sent. The returned win32 wrapper's set_focus() and the keystroke
    sequence that follows are unchanged."""
    matches: list[int] = []

    def _cb(h, _):
        if win32gui.IsWindowVisible(h) and "Ableton Live" in win32gui.GetWindowText(h):
            matches.append(h)
        return True

    win32gui.EnumWindows(_cb, None)
    if not matches:
        raise RuntimeError("Ableton Live window not found — is Live running?")
    app = Application(backend="win32").connect(handle=matches[0])
    return app.window(handle=matches[0])


def list_windows() -> None:
    print("=== Top-level windows ===")
    for w in _desktop().windows():
        try:
            print(f"  title={w.window_text()!r:50} class={w.class_name()!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  <error reading window: {e}>")


def inspect() -> None:
    """Dump windows and, if open, the Export and Save dialog control trees."""
    list_windows()
    for label, matcher in (
        ("EXPORT dialog", dict(title_re=EXPORT_TITLE_RE)),
        ("SAVE dialog", dict(class_name=SAVE_DIALOG_CLASS)),
    ):
        try:
            dlg = _desktop().window(**matcher)
            if dlg.exists():
                print(f"\n=== {label} controls ===")
                dlg.print_control_identifiers(depth=2)
        except Exception as e:  # noqa: BLE001
            print(f"\n{label}: not found ({e})")


def export_batch(
    batch_dir: str | Path,
    expected_count: int | None = None,
    settle: float = 0.6,
    export_dialog_wait: float = 2.0,
    save_dialog_wait: float = 2.5,
    timeout: float = 120.0,
) -> int:
    """Run one export of all individual tracks into ``batch_dir`` using its name as prefix.

    Returns the number of ``*.wav`` files present afterward. ``expected_count`` (if
    given) is how many to wait for before returning.
    """
    batch_dir = Path(batch_dir).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    prefix = batch_dir.name
    save_target = str(batch_dir / prefix)  # Live appends " <track>.wav"

    print(f"Export target: {save_target}", flush=True)
    live = find_live()
    live.set_focus()
    time.sleep(settle)

    # 1) Open Live's Export dialog. It is custom-drawn INSIDE Live (not a native
    #    window), so pywinauto can't click its controls — we drive it by keyboard.
    print("  Ctrl+Shift+R (open Export dialog)…", flush=True)
    send_keys("^+r")
    time.sleep(export_dialog_wait)

    # 2) Press Enter to trigger the default "Export" button -> native Save dialog.
    print("  Enter (trigger Export)…", flush=True)
    send_keys("{ENTER}")
    time.sleep(save_dialog_wait)  # let the Save dialog appear (filename field focused)

    # 3) Drive the Save dialog by keyboard only (pywinauto can't reliably *find* it):
    #    select the current filename, paste our full path, confirm. Pasting a full
    #    absolute path redirects the dialog to our folder + prefix.
    print("  pasting path + Enter…", flush=True)
    _set_clipboard(save_target)
    send_keys("^a")  # select existing filename
    time.sleep(0.3)
    send_keys("^v")  # paste our full path
    time.sleep(0.3)
    send_keys("{ENTER}")  # save

    # 4) Wait for the rendered files (unique batch folder -> no overwrite prompt).
    return _wait_for_files(batch_dir, prefix, expected_count, timeout)


# --- low-level helpers ------------------------------------------------------

def _wait_for_files(
    batch_dir: Path, prefix: str, expected_count: int | None, timeout: float
) -> int:
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        wavs = list(batch_dir.glob(f"{prefix}*.wav"))
        n = len(wavs)
        if n != last:
            print(f"  …{n} file(s) so far", flush=True)
            last = n
        if expected_count is not None and n >= expected_count:
            time.sleep(1.0)  # let the last file finish writing
            return len(list(batch_dir.glob(f"{prefix}*.wav")))
        time.sleep(1.0)
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="Automate Ableton Export Audio.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect", help="dump windows + dialog controls (run with a dialog open)")
    exp = sub.add_parser("export", help="export all individual tracks into a batch folder")
    exp.add_argument("batch_dir")
    exp.add_argument("--expected", type=int, default=None, help="number of files to wait for")
    exp.add_argument("--timeout", type=float, default=120.0)

    args = ap.parse_args()
    if args.cmd == "inspect":
        inspect()
    else:
        n = export_batch(args.batch_dir, expected_count=args.expected, timeout=args.timeout)
        print(f"Done. {n} file(s) in {args.batch_dir}.")


if __name__ == "__main__":
    main()
