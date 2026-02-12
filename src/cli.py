"""CLI entry point powered by Typer."""

import sys
import cv2
from pathlib import Path
from typing import Annotated
import typer
from src.data.loader import Loader
from src.enhancement.postprocess import Postprocess
from src.data.transforms import Transforms
from src.enhancement.upscaler import Upscaler
from src.enhancement.detail_enhancer import DetailEnhancer
from src.enhancement.atmospheric import AtmosphericEnhancer
from src.enhancement.video import VideoEnhancer


def enhance(
    input_path: Annotated[str, typer.Argument(help="Path to RAR archive or input directory")],
    output_path: Annotated[str, typer.Option(help="Output directory for enhanced images")] = "data/enhanced/",
    postprocess: Annotated[bool, typer.Option("--postprocess", help="Apply denoising, sharpening, and color correction")] = False,
    enhance_details: Annotated[bool, typer.Option("--enhance-details", help="Apply CLAHE to enhance fine details (tires, textures, floor)")] = False,
    detail_method: Annotated[str, typer.Option("--detail-method", help="Detail enhancement method: clahe, detail_layer, combined, or text_sharp")] = "clahe",
    detail_strength: Annotated[float, typer.Option("--detail-strength", help="Detail enhancement strength (1.0 = default)")] = 1.0,
    atmospheric: Annotated[bool, typer.Option("--atmospheric", help="Apply atmospheric/eerie color grading")] = False,
    blur_strength: Annotated[float, typer.Option("--blur-strength", help="Atmospheric blur strength (0-100)")] = 47,
    haze: Annotated[float, typer.Option("--haze", help="Haze/fog effect (0-100)")] = 17,
    temp: Annotated[float, typer.Option("--temp", help="Color temperature (-50=cool/blue, +50=warm/orange)")] = -8,
    tint: Annotated[float, typer.Option("--tint", help="Color tint (-50=magenta, +50=green)")] = 30,
    saturation: Annotated[float, typer.Option("--saturation", help="Saturation adjustment (-50 to +50)")] = 13,
    brightness: Annotated[float, typer.Option("--brightness", help="Brightness adjustment (-50 to +50)")] = 1,
    grain: Annotated[float, typer.Option("--grain", help="Film grain/noise amount (0-100)")] = 95,
    video: Annotated[bool, typer.Option("--video", help="Run the video enhancement path")] = False,
    preserve_audio: Annotated[bool, typer.Option("--preserve-audio", help="Reattach original audio (video only)")] = True,
):
    """Run pixel enhancement or FMV enhancement using shared options."""

    if video:
        atm_opts = {
            "blur_strength": blur_strength,
            "haze": haze,
            "temp": temp,
            "tint": tint,
            "saturation": saturation,
            "brightness": brightness,
            "grain": grain,
        }
        enhancer = VideoEnhancer(
            model_name="RealESRGAN_x4plus",
            detail_method=detail_method,
            detail_strength=detail_strength,
            atm_opts=atm_opts,
        )
        input_path_obj = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_video = output_dir / f"{input_path_obj.stem}_enhanced.mp4"
        typer.echo(f"Enhancing video {input_path} → {final_video} (audio preserved={preserve_audio})")
        enhancer.enhance_file(input_path, str(final_video), preserve_audio=preserve_audio)
        typer.echo("Video enhancement complete.")
        return

    loader = Loader(input_path, "data/input/")
    files = loader.extract_images()

    post = Postprocess("data/input/", "data/postprocessed/") if postprocess else None
    transforms = Transforms("data/transformed/")
    output_root = Path(output_path)
    batch_output = output_root / Path(input_path).stem
    batch_output.mkdir(parents=True, exist_ok=True)
    upscaler = Upscaler("RealESRGAN_x4plus", batch_output)
    detail_enhancer = DetailEnhancer(batch_output) if enhance_details else None
    atmospheric_enhancer = AtmosphericEnhancer(batch_output) if atmospheric else None

    mode_str = "without"
    if postprocess:
        mode_str = "with"
    detail_str = ""
    if enhance_details:
        detail_str = f" + {detail_method} detail enhancement"
    atmo_str = ""
    if atmospheric:
        atmo_str = " + eerie atmospheric grading"

    typer.echo(f"Processing {len(files)} images {mode_str} postprocessing{detail_str}{atmo_str}...")

    for idx, img_path in enumerate(files, 1):
        typer.echo(f"[{idx}/{len(files)}] Processing: {img_path.name}")
        img = cv2.imread(str(img_path))

        if postprocess and post:
            typer.echo("  → Applying postprocessing...")
            img = post.apply_all(img)

        tensor = transforms(img)
        enhanced_tensor = upscaler.upscale_tensor(tensor)
        enhanced_img = upscaler._tensor_to_uint8(enhanced_tensor)

        if enhance_details and detail_enhancer:
            typer.echo(f"  → Enhancing details with {detail_method}...")
            enhanced_img = detail_enhancer.enhance(
                enhanced_img,
                method=detail_method,
                strength=detail_strength,
            )

        if atmospheric and atmospheric_enhancer:
            typer.echo("  → Applying atmospheric color grading...")
            enhanced_img = atmospheric_enhancer.apply_eerie_atmosphere(
                enhanced_img,
                blur_strength=blur_strength,
                haze=haze,
                temp=temp,
                tint=tint,
                saturation=saturation,
                brightness=brightness,
                grain=grain,
            )

        upscaler.save_enhanced(enhanced_img, Path(img_path).stem)


def main():
    _normalize_sys_argv()
    typer.run(enhance)


if __name__ == "__main__":
    main()


def _normalize_sys_argv():
    if len(sys.argv) <= 1:
        return
    rest = sys.argv[1:]
    path_parts = []
    idx = 0
    while idx < len(rest) and not rest[idx].startswith("--"):
        path_parts.append(rest[idx])
        idx += 1
    if not path_parts:
        return
    combined = " ".join(path_parts).strip()
    rest = [combined] + rest[idx:]
    sys.argv = [sys.argv[0]] + rest
