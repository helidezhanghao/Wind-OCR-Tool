import unittest

from coordinate_utils import get_utm_zone, utm_to_wgs84, wgs84_to_utm


class UTMConversionTests(unittest.TestCase):
    def assert_roundtrip(self, latitude, longitude, expected_zone, expected_hemi):
        east, north, zone, hemi, _, epsg = wgs84_to_utm(latitude, longitude)
        self.assertIsNotNone(east)
        self.assertEqual(zone, expected_zone)
        self.assertEqual(hemi, expected_hemi)
        self.assertEqual(epsg, (32600 if expected_hemi == "N" else 32700) + expected_zone)

        result_lat, result_lon = utm_to_wgs84(east, north, zone, hemi)
        self.assertAlmostEqual(result_lat, latitude, places=6)
        self.assertAlmostEqual(result_lon, longitude, places=6)

    def test_beijing_roundtrip(self):
        self.assert_roundtrip(39.9042, 116.4074, 50, "N")

    def test_sydney_roundtrip(self):
        self.assert_roundtrip(-33.8688, 151.2093, 56, "S")

    def test_manual_zone_override(self):
        result = wgs84_to_utm(39.9042, 116.4074, zone_override=49)
        self.assertEqual(result[2], 49)
        self.assertEqual(result[5], 32649)

    def test_zone_boundaries(self):
        self.assertEqual(get_utm_zone(-180), 1)
        self.assertEqual(get_utm_zone(180), 60)

    def test_invalid_inputs_return_none(self):
        self.assertEqual(wgs84_to_utm(95, 116), (None,) * 6)
        self.assertEqual(wgs84_to_utm(39, 116, hemi_override="S"), (None,) * 6)
        self.assertEqual(utm_to_wgs84(500000, 4000000, 0, "N"), (None, None))
        self.assertEqual(utm_to_wgs84(-1, 4000000, 50, "N"), (None, None))


if __name__ == "__main__":
    unittest.main()
