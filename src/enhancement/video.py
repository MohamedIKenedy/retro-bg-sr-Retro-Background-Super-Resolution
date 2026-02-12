import cv2
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from .upscaler import Upscaler
from .detail_enhancer import DetailEnhancer
from .atmospheric import AtmosphericEnhancer


class VideoEnhancer:
    """In-memory video enhancer: reads frames via OpenCV, processes them in RAM,
    and writes a final video file. Optionally reattaches original audio.
    """

    def __init__(self,
                 model_name: str = "RealESRGAN_x4plus",
                 detail_method: str = "combined",
                 detail_strength: float = 1.0,
                 atm_opts: Optional[Dict] = None):
        self.upscaler = Upscaler(model_name)
        self.detail = DetailEnhancer()
        self.atm = AtmosphericEnhancer()
        self.detail_method = detail_method
        self.detail_strength = detail_strength
        self.atm_opts = atm_opts or {}

    def _process_frame(self, frame):
        """Process a single BGR uint8 frame and return processed BGR uint8 frame."""
        # Upscale using public helper
        up_img = self.upscaler.upscale_array(frame)

        # Detail enhancement
        up_img = self.detail.enhance(up_img, method=self.detail_method, strength=self.detail_strength)

        # Atmospheric
        atm_img = self.atm.apply_eerie_atmosphere(
            up_img,
            blur_strength=self.atm_opts.get("blur_strength", 0),
            haze=self.atm_opts.get("haze", 8),
            temp=self.atm_opts.get("temp", -12),
            tint=self.atm_opts.get("tint", 5),
            saturation=self.atm_opts.get("saturation", -3),
            brightness=self.atm_opts.get("brightness", 0),
            contrast=self.atm_opts.get("contrast", 5),
            grain=self.atm_opts.get("grain", 10),
            fog_color=self.atm_opts.get("fog_color", (100, 115, 105)),
        )

        return atm_img

    def enhance_file(self, input_video: str, output_video: str, preserve_audio: bool = True):
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = getattr(self.upscaler, "scale", 1)
        out_size = (w * scale, h * scale)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        tmp_out = Path(tempfile.gettempdir()) / (Path(output_video).stem + "_noaudio.mp4")
        writer = cv2.VideoWriter(str(tmp_out), fourcc, fps, out_size)

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                processed = self._process_frame(frame)
                writer.write(processed)
        finally:
            cap.release()
            writer.release()

        if preserve_audio:
            final_out = Path(output_video)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(tmp_out),
                "-i", str(input_video),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                str(final_out)
            ]
            subprocess.run(cmd, check=True)
            try:
                tmp_out.unlink()
            except Exception:
                pass
        else:
            Path(tmp_out).replace(output_video)
