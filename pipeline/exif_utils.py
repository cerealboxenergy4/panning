"""Small EXIF helpers shared by the panning pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, ImageOps

EXIF_BY_NAME = {name: tag for tag, name in ExifTags.TAGS.items()}
DEFAULT_FOCAL_MM = 31.0
DEFAULT_SENSOR_W_MM = 23.5
DEFAULT_IMAGE_W_PX = 6000
DEFAULT_EXPOSURE_S = 1.0 / 125.0


def rational_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, tuple) and len(value) == 2 and value[1] != 0:
        return float(value[0]) / float(value[1])
    return None


def _iter_ifds(exif):
    yield exif
    if not hasattr(exif, 'get_ifd'):
        return

    ifd_ids = [0x8769, 0x8825, 0xA005]  # Exif, GPS, Interop
    if hasattr(ExifTags, 'IFD'):
        for attr in ['Exif', 'GPSInfo', 'Interop']:
            if hasattr(ExifTags.IFD, attr):
                ifd_ids.append(getattr(ExifTags.IFD, attr))

    seen = set()
    for ifd_id in ifd_ids:
        key = str(ifd_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            nested = exif.get_ifd(ifd_id)
        except Exception:
            continue
        if nested:
            yield nested


def get_exif_value(exif, name: str):
    tag = EXIF_BY_NAME.get(name)
    if tag is None:
        return None
    for ifd in _iter_ifds(exif):
        value = ifd.get(tag)
        if value is not None:
            return value
    return None


def iso_from_exif(exif) -> tuple[float | None, str | None]:
    for name in ['PhotographicSensitivity', 'ISOSpeedRatings', 'RecommendedExposureIndex']:
        value = get_exif_value(exif, name)
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        iso = rational_to_float(value)
        if iso and iso > 0:
            return iso, f'exif_{name}'
    return None, None


def aperture_from_exif(exif) -> tuple[float | None, str | None]:
    f_number = rational_to_float(get_exif_value(exif, 'FNumber'))
    if f_number and f_number > 0:
        return f_number, 'exif_f_number'

    aperture_value = rational_to_float(get_exif_value(exif, 'ApertureValue'))
    if aperture_value is not None:
        f_number = 2.0 ** (aperture_value / 2.0)
        if f_number > 0:
            return f_number, 'exif_aperture_value'
    return None, None


def photometric_exposure_value(exposure_s: float | None, iso: float | None, f_number: float | None) -> float | None:
    if exposure_s is None or iso is None or f_number is None or f_number <= 0:
        return None
    return float(exposure_s) * float(iso) / (float(f_number) ** 2)


def _json_value(value: Any):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    converted = rational_to_float(value)
    if converted is not None:
        return converted
    return str(value)


def exposure_from_exif(exif) -> tuple[float | None, str | None]:
    exposure = rational_to_float(get_exif_value(exif, 'ExposureTime'))
    if exposure and exposure > 0:
        return exposure, 'exif_exposure_time'

    shutter_apex = rational_to_float(get_exif_value(exif, 'ShutterSpeedValue'))
    if shutter_apex is not None:
        exposure = 2.0 ** (-shutter_apex)
        if exposure > 0:
            return exposure, 'exif_shutter_speed_value'
    return None, None


def read_image_exif_metadata(path: str | Path, sensor_width_mm: float | None = None) -> dict[str, Any]:
    path = Path(path)
    image = Image.open(path)
    oriented = ImageOps.exif_transpose(image)
    exif = image.getexif()
    width_px, height_px = oriented.size

    focal_mm = rational_to_float(get_exif_value(exif, 'FocalLength'))
    focal_35 = rational_to_float(get_exif_value(exif, 'FocalLengthIn35mmFilm'))
    fp_x = rational_to_float(get_exif_value(exif, 'FocalPlaneXResolution'))
    fp_unit = get_exif_value(exif, 'FocalPlaneResolutionUnit')
    exposure_s, exposure_source = exposure_from_exif(exif)
    iso, iso_source = iso_from_exif(exif)
    f_number, f_number_source = aperture_from_exif(exif)
    exposure_value = photometric_exposure_value(exposure_s, iso, f_number)

    meta: dict[str, Any] = {
        'path': str(path),
        'width_px': int(width_px),
        'height_px': int(height_px),
        'make': _json_value(get_exif_value(exif, 'Make')),
        'model': _json_value(get_exif_value(exif, 'Model')),
        'lens_model': _json_value(get_exif_value(exif, 'LensModel')),
        'orientation': _json_value(get_exif_value(exif, 'Orientation')),
        'datetime_original': _json_value(get_exif_value(exif, 'DateTimeOriginal')),
        'exif_focal_mm': focal_mm,
        'exif_focal_35mm': focal_35,
        'exif_exposure_s': exposure_s,
        'exif_exposure_source': exposure_source,
        'exif_iso': iso,
        'exif_iso_source': iso_source,
        'exif_f_number': f_number,
        'exif_f_number_source': f_number_source,
        'exif_photometric_exposure_value': exposure_value,
        'exif_focal_plane_x_resolution': fp_x,
        'exif_focal_plane_resolution_unit': _json_value(fp_unit),
        'focal_px': None,
        'focal_source': None,
        'pixel_pitch_mm': None,
        'sensor_width_mm': None,
    }

    unit_to_mm = {
        2: 25.4,  # inch
        3: 10.0,  # centimeter
        4: 1.0,   # millimeter
        5: 0.001, # micrometer
    }
    if focal_mm and fp_x and fp_unit in unit_to_mm:
        pixel_pitch_mm = unit_to_mm[fp_unit] / fp_x
        meta.update({
            'focal_px': float(focal_mm / pixel_pitch_mm),
            'focal_source': 'exif_focal_length_and_focal_plane_resolution',
            'pixel_pitch_mm': float(pixel_pitch_mm),
            'sensor_width_mm': float(pixel_pitch_mm * width_px),
        })
    elif focal_35:
        meta.update({
            'focal_px': float(width_px * focal_35 / 36.0),
            'focal_source': 'exif_35mm_equivalent',
            'sensor_width_mm': 36.0,
        })
    elif focal_mm and sensor_width_mm:
        pixel_pitch_mm = float(sensor_width_mm) / float(width_px)
        meta.update({
            'focal_px': float(focal_mm / pixel_pitch_mm),
            'focal_source': 'exif_focal_length_with_sensor_width_arg',
            'pixel_pitch_mm': pixel_pitch_mm,
            'sensor_width_mm': float(sensor_width_mm),
        })

    missing = []
    if meta['focal_px'] is None:
        missing.append('focal_px')
    if exposure_s is None:
        missing.append('exposure_s')
    if iso is None:
        missing.append('iso')
    if f_number is None:
        missing.append('f_number')
    meta['missing_standard_fields'] = missing
    return meta


def resolve_camera_calibration(
    image_path: str | Path,
    focal_px: float | None = None,
    focal_mm: float | None = None,
    sensor_width_mm: float | None = None,
    exposure_s: float | None = None,
    fallback_focal_mm: float = DEFAULT_FOCAL_MM,
    fallback_sensor_width_mm: float = DEFAULT_SENSOR_W_MM,
    fallback_image_width_px: int | None = None,
    fallback_exposure_s: float = DEFAULT_EXPOSURE_S,
) -> dict[str, Any]:
    exif_meta = read_image_exif_metadata(image_path, sensor_width_mm=sensor_width_mm)
    width_px = int(exif_meta.get('width_px') or fallback_image_width_px or DEFAULT_IMAGE_W_PX)
    fallback_image_width_px = int(fallback_image_width_px or width_px)

    focal_source = None
    used_focal_mm = focal_mm if focal_mm is not None else exif_meta.get('exif_focal_mm')
    used_sensor_width_mm = sensor_width_mm if sensor_width_mm is not None else exif_meta.get('sensor_width_mm')
    pixel_pitch_mm = None

    if focal_px is not None:
        used_focal_px = float(focal_px)
        focal_source = 'argument_focal_px'
        if focal_mm is not None:
            pixel_pitch_mm = float(focal_mm) / used_focal_px
        elif sensor_width_mm is not None:
            pixel_pitch_mm = float(sensor_width_mm) / width_px
    elif focal_mm is not None:
        used_sensor_width_mm = sensor_width_mm or exif_meta.get('sensor_width_mm') or fallback_sensor_width_mm
        pixel_pitch_mm = float(used_sensor_width_mm) / width_px
        used_focal_px = float(focal_mm) / pixel_pitch_mm
        focal_source = 'argument_focal_mm_with_sensor_width'
    elif exif_meta.get('focal_px') is not None:
        used_focal_px = float(exif_meta['focal_px'])
        focal_source = str(exif_meta['focal_source'])
        used_sensor_width_mm = exif_meta.get('sensor_width_mm')
        pixel_pitch_mm = exif_meta.get('pixel_pitch_mm')
    else:
        used_sensor_width_mm = sensor_width_mm or fallback_sensor_width_mm
        used_focal_mm = fallback_focal_mm
        pixel_pitch_mm = float(used_sensor_width_mm) / float(fallback_image_width_px)
        used_focal_px = float(used_focal_mm) / pixel_pitch_mm
        focal_source = 'fallback_focal_mm_sensor_width'

    if pixel_pitch_mm is None and used_sensor_width_mm is not None:
        pixel_pitch_mm = float(used_sensor_width_mm) / width_px
    if used_focal_mm is None and pixel_pitch_mm is not None:
        used_focal_mm = float(used_focal_px) * float(pixel_pitch_mm)

    if exposure_s is not None:
        used_exposure_s = float(exposure_s)
        exposure_source = 'argument_exposure_s'
    elif exif_meta.get('exif_exposure_s') is not None:
        used_exposure_s = float(exif_meta['exif_exposure_s'])
        exposure_source = str(exif_meta.get('exif_exposure_source') or 'exif')
    else:
        used_exposure_s = float(fallback_exposure_s)
        exposure_source = 'fallback_exposure_s'

    return {
        'focal_px': float(used_focal_px),
        'focal_mm': None if used_focal_mm is None else float(used_focal_mm),
        'sensor_width_mm': None if used_sensor_width_mm is None else float(used_sensor_width_mm),
        'pixel_pitch_mm': None if pixel_pitch_mm is None else float(pixel_pitch_mm),
        'image_width_px': int(width_px),
        'image_height_px': int(exif_meta.get('height_px') or 0),
        'exposure_s': float(used_exposure_s),
        'focal_source': focal_source,
        'exposure_source': exposure_source,
        'exif': exif_meta,
    }


def compare_exif_metadata(blurry_meta: dict[str, Any], sharp_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if sharp_meta is None:
        return None

    comparisons: dict[str, Any] = {}
    for field in [
        'focal_px', 'exif_focal_mm', 'exif_focal_35mm', 'exif_exposure_s',
        'exif_iso', 'exif_f_number', 'exif_photometric_exposure_value',
        'sensor_width_mm', 'width_px', 'height_px',
    ]:
        blurry_val = blurry_meta.get(field)
        sharp_val = sharp_meta.get(field)
        entry = {'blurry': blurry_val, 'sharp': sharp_val}
        if isinstance(blurry_val, (int, float)) and isinstance(sharp_val, (int, float)) and blurry_val not in (0, None):
            entry['delta'] = float(sharp_val) - float(blurry_val)
            entry['ratio_sharp_over_blurry'] = float(sharp_val) / float(blurry_val)
        comparisons[field] = entry

    warnings = []
    if blurry_meta.get('focal_px') is None or sharp_meta.get('focal_px') is None:
        warnings.append('standard EXIF focal calibration missing for at least one image')
    if blurry_meta.get('exif_exposure_s') is None or sharp_meta.get('exif_exposure_s') is None:
        warnings.append('standard EXIF exposure missing for at least one image')
    if blurry_meta.get('exif_photometric_exposure_value') is None or sharp_meta.get('exif_photometric_exposure_value') is None:
        warnings.append('EXIF exposure/ISO/aperture photometric scale missing for at least one image')

    return {
        'blurry_path': blurry_meta.get('path'),
        'sharp_path': sharp_meta.get('path'),
        'fields': comparisons,
        'warnings': warnings,
    }
