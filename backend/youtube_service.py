import os
import yt_dlp


def download_youtube_video(url: str, output_path: str) -> str:
    """
    Descarcă videoclipul de pe YouTube în format MP4 și returnează titlul clipului.
    """
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "YouTube Match Clip")
        return title