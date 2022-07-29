import torch
from opt import get_opts

import numpy as np

from einops import rearrange
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from train import NeRFSystem

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays

import warnings
from argparse import Namespace

warnings.filterwarnings("ignore")


class OrbitCamera:
    def __init__(self, W, H, r=5, fovy=50):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.rot = R.from_quat([0, 1, 0, 0])
        self.up = np.array([0, 1, 0], dtype=np.float32)

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 0.001 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


class NeRFGUI:
    def __init__(self, renderer, H=1080, W=1440, radius=2.5, fovy=50):
        self.renderer = renderer
        self.H = H
        self.W = W
        self.radius = radius
        self.fovy = fovy

        self.cam = OrbitCamera(self.W, self.H, r=self.radius, fovy=self.fovy)

        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

        dpg.create_context()
        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()

    def render_nerf(self):
        dpg.set_value("_texture", self.renderer.render_one_pose(self.cam.pose))

    def register_dpg(self):

        ## register texture ##
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ## register window ##
        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=500, height=150):
            # Pose info
            with dpg.collapsing_header(label="Info", default_open=True):
                # pose
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.cam.pose), tag="_log_pose")

        ## register camera handler ##
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            dx = app_data[1]
            dy = app_data[2]
            self.cam.orbit(dx, dy)
            self.need_update = True
            dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.cam.scale(delta)
            self.need_update = True
            dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            dx = app_data[1]
            dy = app_data[2]
            self.cam.pan(dx, dy)
            self.need_update = True
            dpg.set_value("_log_pose", str(self.cam.pose))

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        ## Window name ##
        dpg.create_viewport(
            title="ngp-pl", width=self.W, height=self.H, resizable=False
        )

        ## Avoid scroll bar in the window ##
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        ## Launch the gui ##
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):

        while dpg.is_dearpygui_running():
            self.render_nerf()
            dpg.render_dearpygui_frame()


class RenderGui:
    def __init__(self, ckpt_path, intrinsics, H, W, shift, scale) -> None:
        self.ckp = torch.load(ckpt_path)
        self.intrinsics = intrinsics
        self.H = H
        self.W = W
        self.shift = shift
        self.scale = scale

        del self.ckp["state_dict"]["poses"]
        del self.ckp["state_dict"]["directions"]
        self.ckp["hyper_parameters"]["eval_lpips"] = False

        # Load checkpoint
        self.system = NeRFSystem(Namespace(**self.ckp["hyper_parameters"])).cuda()
        self.system.load_state_dict(self.ckp["state_dict"])

        # Rays direction
        self.directions = get_ray_directions(self.H, self.W, self.intrinsics)

    def render_one_pose(self, pose):
        rays_o, rays_d = self.get_rays(pose)
        results = self.system(rays_o, rays_d, split="render")

        rgb_pred = rearrange(results["rgb"].cpu().numpy(), "(h w) c -> h w c", h=self.H)

        return rgb_pred

    def get_rays(self, pose):
        pose = pose[:3]
        pose[:, 3] -= self.shift
        pose[:, 3] /= self.scale

        rays_o, rays_d = get_rays(self.directions, torch.cuda.FloatTensor(pose))

        return rays_o, rays_d


if __name__ == "__main__":
    hparams = get_opts()
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {
        "root_dir": hparams.root_dir,
        "downsample": hparams.downsample,
    }
    dataset = dataset(split="val", **kwargs)

    shift = dataset.shift
    scale = dataset.scale

    intrinsics = dataset.K
    w, h = dataset.img_wh

    render_gui = RenderGui(hparams.ckpt_path, intrinsics, h, w, shift, scale)
    gui = NeRFGUI(render_gui, h, w)
    gui.render()
