"""Small, reusable WGS84/UTM conversion helpers."""

from math import isfinite

from pyproj import CRS, Transformer


def get_utm_zone(longitude):
    """Return the standard UTM zone (1-60) for a longitude."""
    lon = float(longitude)
    if not isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError("经度必须在 -180 到 180 之间")
    return max(1, min(60, int((lon + 180) / 6) + 1))


def wgs84_to_utm(lat, lon, zone_override=0, hemi_override=None):
    """Convert WGS84 latitude/longitude to UTM.

    Returns ``(easting, northing, zone, hemisphere, central_meridian, epsg)``.
    Invalid input keeps the app's existing failure contract and returns six
    ``None`` values.
    """
    try:
        latitude = float(lat)
        longitude = float(lon)
        if not all(isfinite(value) for value in (latitude, longitude)):
            raise ValueError
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError

        zone = int(zone_override) if zone_override else get_utm_zone(longitude)
        if not 1 <= zone <= 60:
            raise ValueError

        override = str(hemi_override).upper() if hemi_override else None
        hemisphere = override if override in ("N", "S") else ("N" if latitude >= 0 else "S")
        if (hemisphere == "N" and latitude < 0) or (hemisphere == "S" and latitude > 0):
            raise ValueError

        epsg = (32600 if hemisphere == "N" else 32700) + zone
        central_meridian = zone * 6 - 183
        transformer = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True
        )
        easting, northing = transformer.transform(longitude, latitude)
        if not all(isfinite(value) for value in (easting, northing)):
            raise ValueError
        return easting, northing, zone, hemisphere, central_meridian, epsg
    except (TypeError, ValueError, OverflowError):
        return None, None, None, None, None, None


def utm_to_wgs84(easting, northing, zone, hemi="N"):
    """Convert UTM coordinates to WGS84 latitude/longitude."""
    try:
        east = float(easting)
        north = float(northing)
        zone_number = int(zone)
        hemisphere = str(hemi).upper()

        if not all(isfinite(value) for value in (east, north)):
            raise ValueError
        if not 1 <= zone_number <= 60 or hemisphere not in ("N", "S"):
            raise ValueError
        if not 0 < east < 1_000_000 or not 0 <= north <= 10_000_000:
            raise ValueError

        epsg = (32600 if hemisphere == "N" else 32700) + zone_number
        transformer = Transformer.from_crs(
            CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True
        )
        longitude, latitude = transformer.transform(east, north)
        if not all(isfinite(value) for value in (latitude, longitude)):
            raise ValueError
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError
        return latitude, longitude
    except (TypeError, ValueError, OverflowError):
        return None, None
