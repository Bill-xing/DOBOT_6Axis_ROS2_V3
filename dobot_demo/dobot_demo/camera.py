from pyorbbecsdk import *
import numpy as np
import cv2

from typing import Union, Any, Optional

delta_dict = {
    "2e742906": [0, -2, 10],
    "26fef4210": [0, 0, -2],
}

ctx = Context()
device_list = ctx.query_devices()
print(device_list)

def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    yuyv = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return bgr_image

def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    uyvy = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image

def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image

class Camera():
    def __init__(self, device: Device):
        self.pipeline = Pipeline(device)
        self.device_uid = device.get_device_info().get_uid()
        self.config = Config()

        # color stream config
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = profile_list.get_default_video_stream_profile()
        self.config.enable_stream(self.color_profile)
        # depth stream config
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.depth_profile = profile_list.get_video_stream_profile(800, 600, OBFormat.Y16, 30)
        self.config.enable_stream(self.depth_profile)
        # set color and depth align mode
        self.config.set_align_mode(OBAlignMode.SW_MODE)
        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)

        self.camera_intrinsic = self.get_camera_intrinsic()
        self.camera_extrinsic = self.get_camera_extrinsic()
        print(self.camera_extrinsic)
        self.inverse_K = np.array([
            [1 / self.camera_intrinsic[0], 0, -self.camera_intrinsic[2] / self.camera_intrinsic[0]],
            [0, 1 / self.camera_intrinsic[1], -self.camera_intrinsic[3] / self.camera_intrinsic[1]],
            [0, 0, 1],
        ], dtype = float)
        print(self.inverse_K)
    

    def get_frame_data(self):
        max_attempts = 0
        while(max_attempts < 5):
            max_attempts += 1
            frames: FrameSet = self.pipeline.wait_for_frames(2000)
            color_frame = frames.get_color_frame()
            if color_frame == None: continue
            depth_frame = frames.get_depth_frame()
            if depth_frame == None: continue

            color_image = frame_to_bgr_image(color_frame)
            
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            break

        if max_attempts >= 5: return None, None
        return color_image, depth_data


    def get_camera_intrinsic(self):
        camera_param = self.pipeline.get_camera_param()
        print([
            camera_param.depth_intrinsic.fx, camera_param.depth_intrinsic.fy,
            camera_param.depth_intrinsic.cx, camera_param.depth_intrinsic.cy
        ])
        return [
            camera_param.depth_intrinsic.fx, camera_param.depth_intrinsic.fy,
            camera_param.depth_intrinsic.cx, camera_param.depth_intrinsic.cy
        ]


    def pixel_to_camera_coordinates(self, xy_points: np.array, depth_value: np.array):
        z_axis = np.ones((xy_points.shape[0], 1))
        xyz_points = np.concatenate((xy_points, z_axis), axis = 1)
        depth_value = depth_value[:, np.newaxis]

        new_points = np.dot(self.inverse_K, xyz_points.T).T
        new_points *= depth_value

        return new_points
    

    def get_camera_extrinsic(self) -> np.array:
        try:
            camera_point = []
            real_point = []
            import pickle
            with open('./calibrate_points/{}_camera_points.pkl'.format(self.device_uid), 'rb') as file:
                camera_point = pickle.load(file)
            with open('./calibrate_points/{}_world_points.pkl'.format(self.device_uid), 'rb') as file:
                real_point = pickle.load(file)
            camera_point = np.array(camera_point)
            real_point = np.array(real_point)
            print(len(camera_point))
            print(len(real_point))
            valid_indices = ~np.isnan(camera_point).any(axis = 1)
            camera_point = camera_point[valid_indices]
            real_point = real_point[valid_indices]
            camera_extrinsic = np.linalg.lstsq(camera_point, real_point, rcond = None)[0]
            return camera_extrinsic
        except:
            print("no camera extrinsic, please calibrate!")


    def pixel_to_world_coordinates(self, xy_points: np.array, depth_value: np.array):
        camera_coords = self.pixel_to_camera_coordinates(xy_points, depth_value)
        camera_coords = np.concatenate((camera_coords, np.ones((camera_coords.shape[0], 1))), axis = 1)
        # print(camera_coords)
        world_coords = np.round(np.dot(camera_coords, self.camera_extrinsic), 3)
        world_points = world_coords[:, :3] + np.array(delta_dict[self.device_uid])
        # print(world_points)
        return world_points