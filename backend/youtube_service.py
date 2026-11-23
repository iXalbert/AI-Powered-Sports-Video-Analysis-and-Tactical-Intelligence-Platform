import os
import yt_dlp


def download_youtube_video(url: str, output_path: str, browser: str = "chrome") -> str:
    """
    Descarcă un videoclip YouTube în format MP4 folosind cookies din browser
    pentru a evita eroarea 'Sign in to confirm you’re not a bot'.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
    }

    # Încercăm pe rând browserele uzuale de pe macOS (Chrome, Safari, Brave, Firefox)
    browsers_to_try = [browser, "safari", "brave", "firefox", None]
    last_error = None

    for b in dict.fromkeys(browsers_to_try):
        opts = dict(base_opts)
        if b is not None:
            opts["cookiesfrombrowser"] = (b,)

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return info.get("title", "YouTube Match Clip")
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Nu s-a putut descărca clipul YouTube. Asigură-te că ești logat în YouTube în Chrome/Safari. Detalii: {last_error}"
    )