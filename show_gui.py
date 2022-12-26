import torch
from opt import get_opts
import numpy as np
from einops import rearrange
from scipy.spatial.transform import Rotation as R
import time

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays
from models.networks import NGP
from models.rendering import render
from train import depth2img
from utils import load_ckpt

import warnings; warnings.filterwarnings("ignore")

import taichi as ti

@ti.kernel
def write_buffer(W:ti.i32, H:ti.i32, x: ti.types.ndarray(), final_pixel:ti.template()):
    for i, j in ti.ndrange(W, H):
        for p in ti.static(range(3)):
            final_pixel[i, j][p] = x[H-j, i, p]

class OrbitCamera:
    def __init__(self, K, img_wh, poses, r):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r
        self.center = np.zeros(3)

        pose_np = poses.cpu().numpy()
        # choose a pose as the initial rotation
        self.rot = pose_np[0][:3, :3]

        self.rotate_speed = 0.8
        self.res_defalut = pose_np[0]

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def reset(self, pose=None):
        self.rot = np.eye(3)
        self.center = np.zeros(3)
        self.radius = 2.0
        if pose is not None:
            self.rot = pose.cpu().numpy()[:3, :3]

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(100*self.rotate_speed  * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-100*self.rotate_speed  * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NGPGUI:
    def __init__(self, hparams, K, img_wh, poses, radius=2.5):
        self.hparams = hparams
        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=hparams.scale, rgb_act=rgb_act).cuda()
        load_ckpt(self.model, hparams.ckpt_path)

        self.poses = poses

        self.cam = OrbitCamera(K, img_wh, poses, r=radius)
        self.W, self.H = img_wh
        # self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.render_buffer = ti.Vector.field(3, dtype=float, shape=(self.W, self.H))

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0
        self.exposure = 0.2

    def render_cam(self):
        t = time.time()
        directions = get_ray_directions(self.cam.H, self.cam.W, self.cam.K, device='cuda')
        rays_o, rays_d = get_rays(directions, torch.cuda.FloatTensor(self.cam.pose))

        # TODO: set these attributes by gui
        if self.hparams.dataset_name in ['colmap', 'nerfpp']:
            exp_step_factor = 1/256
        else: exp_step_factor = 0

        results = render(self.model, rays_o, rays_d,
                         **{'test_time': True,
                            'to_cpu': False, 'to_numpy': False,
                            'T_threshold': 1e-2,
                            'exposure': torch.cuda.FloatTensor([self.exposure]),
                            'max_samples': 100,
                            'exp_step_factor': exp_step_factor})

        rgb = rearrange(results["rgb"], "(h w) c -> h w c", h=self.H)
        depth = rearrange(results["depth"], "(h w) -> h w", h=self.H)
        # torch.cuda.synchronize()
        self.dt = time.time()-t
        self.mean_samples = results['total_samples']/len(rays_o)

        if self.img_mode == 0:
            return rgb
        elif self.img_mode == 1:
            return depth2img(depth.cpu().numpy()).astype(np.float32)/255.0

    def check_cam_rotate(self, window, last_orbit_x, last_orbit_y):
        if window.is_pressed(ti.ui.RMB):
            curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
            if last_orbit_x is None or last_orbit_y is None:
                last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
            else:
                dx = curr_mouse_x - last_orbit_x
                dy = curr_mouse_y - last_orbit_y
                self.cam.orbit(dx, -dy)
                last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
        else:
            last_orbit_x = None
            last_orbit_y = None

        return last_orbit_x, last_orbit_y

    def check_key_press(self, window):
        if window.is_pressed('w'):
            self.cam.scale(0.2)
        if window.is_pressed('s'):
            self.cam.scale(-0.2)
        if window.is_pressed('a'):
            self.cam.pan(100, 0.)
        if window.is_pressed('d'):
            self.cam.pan(-100, 0.)
        if window.is_pressed('e'):
            self.cam.pan(0., -100)
        if window.is_pressed('q'):
            self.cam.pan(0., 100)

    def render(self):

        window = ti.ui.Window('ngp_pl', (self.W, self.H),)
        canvas = window.get_canvas()
        gui = window.get_gui()

        # GUI controls variables
        last_orbit_x = None
        last_orbit_y = None

        view_id = 0
        last_view_id = 0

        views_size = self.poses.shape[0]-1

        while window.running:
            self.check_key_press(window)
            last_orbit_x, last_orbit_y = self.check_cam_rotate(window, last_orbit_x, last_orbit_y)
            with gui.sub_window("Control", 0.01, 0.01, 0.4, 0.2) as w:
                self.cam.rotate_speed = w.slider_float('rotate speed', self.cam.rotate_speed, 0.1, 1.)
                self.exposure = w.slider_float('exposure', self.exposure, 1/60, 32)

                self.img_mode = w.checkbox("show depth", self.img_mode)

                view_id = w.slider_int('train view', view_id, 0, views_size)

                if last_view_id != view_id:
                    last_view_id = view_id
                    self.cam.reset(self.poses[view_id])

                w.text(f'samples per rays: {self.mean_samples:.2f} s/r')
                w.text(f'render times: {1000*self.dt:.2f} ms')

            ngp_buffer = self.render_cam()
            write_buffer(self.W, self.H, ngp_buffer, self.render_buffer)
            canvas.set_image(self.render_buffer)
            window.show()


if __name__ == "__main__":
    ti.init(arch=ti.cuda)

    hparams = get_opts()
    kwargs = {'root_dir': hparams.root_dir,
              'downsample': hparams.downsample,
              'read_meta': True}
    dataset = dataset_dict[hparams.dataset_name](**kwargs)

    NGPGUI(hparams, dataset.K, dataset.img_wh, dataset.poses).render()
