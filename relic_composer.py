"""Thin wrapper for the composer package."""
from composer import (
    load_image,
    save_image,
    generate_depth,
    depth_to_normal,
    generate_alpha,
    decompose_albedo_shading,
    process_image,
    build_arg_parser,
    main,
)

__all__ = [
    "load_image",
    "save_image",
    "generate_depth",
    "depth_to_normal",
    "generate_alpha",
    "decompose_albedo_shading",
    "process_image",
    "build_arg_parser",
    "main",
]

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
