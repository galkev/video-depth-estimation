from mat4py import loadmat
import numpy as np


# https://www.codefull.org/2016/03/align-depth-and-color-frames-depth-and-rgb-registration/
# https://de.mathworks.com/help/vision/ug/camera-calibration.html
class DepthRegistration(object):
    def __init__(self, f_rgb, f_d, pp_rgb, pp_d, kc_rgb, kc_d, ext_transform, rot_depth=None):
        self.f_rgb = f_rgb  # 2d vec focal lengths
        self.f_d = f_d  # 2d vec focal lengths
        self.pp_rgb = pp_rgb  # 2d vec principal points
        self.pp_d = pp_d  # 2d vec principal points
        self.kc_rgb = kc_rgb  # 5d brown distortion coefficients
        self.kc_d = kc_d  # 5d brown distortion coefficients

        self.ext_transform = ext_transform  # 4x4 transform mat
        self.rot_depth = rot_depth

    #  https://www.codefull.org/2016/03/align-depth-and-color-frames-depth-and-rgb-registration/
    #  http://nicolas.burrus.name/index.php/Research/KinectCalibration
    #  https://github.com/IntelRealSense/librealsense/blob/master/include/librealsense2/rsutil.h
    @staticmethod
    def deproject_pixel_to_point(pixel, depth, f, pp, kc):
        x = (pixel[0] - pp[0]) / f[0]
        y = (pixel[1] - pp[1]) / f[1]

        if kc is not None:
            x, y = DepthRegistration.distortion_brown(x, y, kc, inverse=True)

        point = np.stack([x, y, np.ones(x.shape[0])])

        return point * depth

    @staticmethod
    def transform_point_to_point(point, ext_transform):
        point_hom = np.concatenate([point, np.ones(point.shape[1])[None, :]])
        return np.dot(ext_transform, point_hom)[0:3]

    @staticmethod
    def project_point_to_pixel(point, f, pp, kc):
        x = point[0] / point[2]
        y = point[1] / point[2]

        if kc is not None:
            x, y = DepthRegistration.distortion_brown(x, y, kc, inverse=False)

        return np.array([
            x * f[0] + pp[0],
            y * f[1] + pp[1]
        ])

    @staticmethod
    def distortion_brown(x, y, kc, inverse):
        r2 = x * x + y * y
        f = 1 + kc[0] * r2 + kc[1] * r2 * r2 + kc[4] * r2 * r2 * r2

        if not inverse:
            x *= f
            y *= f
            x_out = x + 2 * kc[2] * x * y + kc[3] * (r2 + 2 * x * x)
            y_out = y + 2 * kc[3] * x * y + kc[2] * (r2 + 2 * y * y)
        else:
            x_out = x * f + 2 * kc[2] * x * y + kc[3] * (r2 + 2 * x * x)
            y_out = y * f + 2 * kc[3] * x * y + kc[2] * (r2 + 2 * y * y)

        return [x_out, y_out]

    def __call__(self, depth_map):
        return self.call_vectorized(depth_map)
        # return self.call_loop(depth_map)

    def test_regist(self, depth_map):
        d1 = self.call_loop(depth_map)
        d2 = self.call_vectorized(depth_map)
        err = np.sum((d1 - d2) ** 2)
        print(err)

    def call_vectorized(self, depth_map):
        if self.rot_depth is not None:
            depth_map = np.rot90(depth_map, k=-self.rot_depth)

        h, w = depth_map.shape
        depth_map_aligned = np.full([h, w], np.inf)

        xx, yy = np.meshgrid(range(w), range(h))
        point = DepthRegistration.deproject_pixel_to_point(
            np.stack([xx.flatten(), yy.flatten()]),
            depth_map[yy.flatten(), xx.flatten()],
            self.f_d,
            self.pp_d,
            self.kc_d
        )

        point_transformed = DepthRegistration.transform_point_to_point(point, self.ext_transform)

        pixel = DepthRegistration.project_point_to_pixel(
            point_transformed,
            self.f_rgb,
            self.pp_rgb,
            self.kc_rgb
        )

        #mask = (pixel[0] >= 0) & (pixel[1] >= 0) & (pixel[0] <= w - 1) & (pixel[1] <= h - 1)
        mask = (pixel[0] >= 0) & (pixel[1] >= 0) & (pixel[0] <= w - 1) & (pixel[1] <= h - 1) #& (point_transformed[2] < 800)

        idx = np.round(pixel[:, mask]).astype(int)
        depths = point_transformed[2, mask]

        #depth_map_aligned[idx[1], idx[0]] = depths

        for i in range(idx.shape[1]):
            #if depths[i] > 0:
            depth_map_aligned[idx[1, i], idx[0, i]] = min(depths[i], depth_map_aligned[idx[1, i], idx[0, i]])

        depth_map_aligned[depth_map_aligned == np.inf] = 0

        depth_map_aligned = depth_map_aligned.reshape(h, w).astype(np.uint16)

        if self.rot_depth is not None:
            depth_map_aligned = np.rot90(depth_map_aligned, k=self.rot_depth)

        return depth_map_aligned

    def call_loop(self, depth_map):
        h, w = depth_map.shape
        depth_map_aligned = np.zeros_like(depth_map)

        for y_d in range(h):
            for x_d in range(w):
                point = DepthRegistration.deproject_pixel_to_point(
                    [x_d, y_d],
                    depth_map[y_d, x_d],
                    self.f_d,
                    self.pp_d,
                    self.kc_d
                )
                point_transformed = DepthRegistration.transform_point_to_point(point, self.ext_transform)
                pixel = DepthRegistration.project_point_to_pixel(
                    point_transformed,
                    self.f_rgb,
                    self.pp_rgb,
                    self.kc_rgb
                )

                if pixel[0] < 0 or pixel[0] < 0 or pixel[0] >= depth_map_aligned.shape[1] or \
                        pixel[1] >= depth_map_aligned.shape[0]:
                    # print("({}, {}) out of bounds".format(p2d_rgb_x, p2d_rgb_y))
                    pass
                else:
                    depth_map_aligned[int(np.round(pixel[1])), int(np.round(pixel[0]))] = point_transformed[2]

        return depth_map_aligned.astype(np.uint16)

    @staticmethod
    def from_matlab_calib(file, undistort=True, rot_depth=None):
        mat = loadmat(file)

        rgb_tag = "left"
        depth_tag = "right"

        # transform in mm
        ext_transform = np.concatenate([
            np.concatenate([mat["R"], mat["T"]], axis=1),
            [[0, 0, 0, 1]]],
            axis=0
        )

        ext_transform = np.linalg.inv(ext_transform)

        return DepthRegistration(
            mat["fc_" + rgb_tag],
            mat["fc_" + depth_tag],
            mat["cc_" + rgb_tag],
            mat["cc_" + depth_tag],
            mat["kc_" + rgb_tag] if undistort else None,
            mat["kc_" + depth_tag] if undistort else None,
            ext_transform,
            rot_depth)

    def __repr__(self):
        return \
            "Focal Length RGB: {}\n" \
            "Focal Length D: {}\n" \
            "Principal Point RGB: {}\n" \
            "Principal Point D: {}\n" \
            "Extrinsics Transform:\n{}".format(
                self.f_rgb,
                self.f_d,
                self.pp_rgb,
                self.pp_d,
                self.ext_transform
            )

    # https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/generate_pointcloud.py
    def save_pointcloud(self, ply_file, rgb, registered_depth, scaling=5000.0):
        center_x = rgb.shape[1] / 2
        center_y = rgb.shape[0] / 2

        points = []
        for v in range(rgb.shape[0]):
            for u in range(rgb.shape[1]):
                color = rgb[v, u]
                z = registered_depth[v, u] / scaling
                if z == 0:
                    continue
                x = (u - center_x) * z / self.f_rgb[0]
                y = (v - center_y) * z / self.f_rgb[1]
                points.append("%f %f %f %d %d %d 0\n" % (x, y, z, color[0], color[1], color[2]))

        file = open(ply_file, "w")
        file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points)))
        file.close()
