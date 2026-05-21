"""Write-stage progress reporting for deterministic aSoCC outputs."""

from pyaesa.shared.runtime.text import USER_TEXT_LINE_WIDTH, compact_user_text


def tick_write_progress(*, context, state) -> None:
    """Advance and print branch-local write progress as a bar and percent."""
    total = int(getattr(state, "write_progress_total", 0) or 0)
    if total <= 0:
        return
    current = int(getattr(state, "write_progress_current", 0) or 0) + 1
    if current > total:
        current = total
    state.write_progress_current = current
    ratio = current / total
    bar_width = 24
    filled = min(bar_width, max(0, int(round(ratio * bar_width))))
    progress_bar = f"[{'#' * filled}{'-' * (bar_width - filled)}]"
    percent = int(round(ratio * 100.0))
    prefix = getattr(state, "write_progress_prefix", None)
    if not isinstance(prefix, str) or not prefix.strip():
        prefix = f"[{context.source}]"
    label = getattr(state, "write_progress_label", None)
    if isinstance(label, str):
        label = compact_user_text(label.strip(), max_chars=48)
    if label:
        line = f"{prefix} writing {label} {progress_bar} {percent:3d}%"
    else:
        line = f"{prefix} writing outputs... {progress_bar} {percent:3d}%"
    line = compact_user_text(line, max_chars=USER_TEXT_LINE_WIDTH)
    progress = getattr(state, "runtime_progress", None)
    log_message = getattr(progress, "log_message", None)
    if callable(log_message):
        log_message(line, persistent=False)
        if current >= total:
            state.write_progress_last_width = 0
        return
    last_width = int(getattr(state, "write_progress_last_width", 0) or 0)
    width = max(last_width, len(line))
    state.write_progress_last_width = width
    print(
        f"\r{line.ljust(width)}",
        end="",
        flush=True,
    )
    if current >= total:
        print()
        state.write_progress_last_width = 0
