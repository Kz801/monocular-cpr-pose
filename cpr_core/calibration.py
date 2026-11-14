class SpatialCalibration:
    def __init__(self, factor=None):
        self.gamma = factor

    def calibrate_ruler(self, p1, p2, physical_length_cm=16.0):
        h_pixel = abs(p1[1] - p2[1])
        if h_pixel <= 0:
            raise ValueError("Zero vertical pixel span")
        self.gamma = float(physical_length_cm) / float(h_pixel)
        return self.gamma

    def calibrate_segmentation(self, pixel_height, physical_height_cm=160.0):
        if pixel_height <= 0:
            raise ValueError("Invalid pixel height")
        self.gamma = float(physical_height_cm) / float(pixel_height)
        return self.gamma

    def pixel_to_cm(self, pixel_val):
        if self.gamma is None:
            raise ValueError("Calibration factor gamma not initialized")
        return pixel_val * self.gamma
