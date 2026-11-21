import shutil
import yt_dlp


def download_youtube_video(url: str, output_path: str) -> str:
    """
    Descarcă videoclipul de pe YouTube în format MP4 și returnează titlul clipului.
    Folosește merge HD dacă ffmpeg este disponibil, altfel flux direct MP4.
    """
    has_ffmpeg = shutil.which("ffmpeg") is not None

    if has_ffmpeg:
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    else:
        fmt = "best[ext=mp4]/best"

    ydl_opts = {
        "format": fmt,
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if has_ffmpeg:
        ydl_opts["merge_output_format"] = "mp4"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "YouTube Match Clip")
        return title