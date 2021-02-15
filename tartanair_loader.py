import sys
import numpy as np
import matplotlib.cm as cm
import open3d as o3d
from PIL import Image
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation
inv = np.linalg.inv


class TartanAirLoader(object):
    RGB_MEAN = np.asarray([101.7402597833251292, 101.2859794623116443,  97.5682397588527266], dtype=np.float32)
    RGB_STD = np.asarray([78.3968977988105991, 77.3794957800464402, 78.6678918708813910], dtype=np.float32)
    H, W = (480, 640)

    def __init__(self, path, max_delta=10):
        self.dataset = {}
        self.trainset = []
        self.testset = []
        for e in sorted(glob(path + '/*/')):
            for d in sorted(glob(e + '/*/')):
                trajectories = sorted(glob(d + '/*/'))
                for i, t in enumerate(trajectories):
                    image = sorted([_ for _ in glob(t + 'image_left/*.png')])
                    depth = sorted([_ for _ in glob(t + 'depth_left/*.npy')])
                    pose = np.loadtxt(t + 'pose_left.txt')

                    key = '_'.join([e.split('/')[-2], d.split('/')[-2], t.split('/')[-2]])
                    data = [{'image': image[j], 'depth':depth[j], 'pose':pose[j]} for j in range(len(image))]

                    self.dataset[key] = data

                    if i < len(trajectories) - 2:
                        self.trainset.append(key)
                    else:
                        self.testset.append(key)

        print(
            f'train_trajectories(total_view): {len(self.trainset)}({sum([len(self.dataset[k]) for k in self.trainset])})')
        print(
            f'test_trajectories(total_view): {len(self.testset)}({sum([len(self.dataset[k]) for k in self.testset])})')

        self.f = self.W / 2
        self.K = np.array([[self.f,      0, self.W / 2],
                           [0,      self.f, self.H / 2],
                           [0,           0,          1]], dtype=np.float32)

        self.max_delta = max_delta

    def load_data(self, trajectory, index):
        data = self.dataset[trajectory][index]
        try:
            image = np.asarray(Image.open(data['image']))
            depth = np.load(data['depth'])
            pose = self.posevec2transformation(data['pose'])
        except:
            raise IOError(f"Couldn\'t load file(s).\n  trajectory: {trajectory}\n  index: {index}\n")

        return image, depth, pose

    def _load_wrapper(self, args):
        return self.load_data(args[0], args[1])

    def get_batch(self, batch_size, test=False):
        trajectories = self.testset if test else self.trainset
        view1 = []
        view2 = []
        for t in np.random.choice(trajectories, batch_size):
            a = np.random.randint(0, len(self.dataset[t]) - self.max_delta)
            b = a + np.random.randint(1, self.max_delta + 1)
            view1.append((t, a))
            view2.append((t, b))

        with ThreadPoolExecutor() as executor:
            data = executor.map(self._load_wrapper, view1 + view2)
        data = np.asarray(list(data))

        x = np.stack(data[:, 0])
        x = np.transpose((x - self.RGB_MEAN) / self.RGB_STD, (0, 3, 1, 2))

        d = np.stack(data[:, 1])
        d[d > 255] = 255

        g = np.stack(data[:, 2])

        return np.split(x, 2), np.split(d, 2), np.split(g, 2)

    def posevec2transformation(self, pose):
        tz, tx, ty = pose[:3]
        qz, qx, qy, qw = pose[3:]

        t = np.asarray([[tx, ty, tz]]).T

        # qxqx, qyqy, qzqz, qwqw = qx * qx, qy * qy, qz * qz, qw * qw
        # qwqx, qwqy, qwqz = qw * qx, qw * qy, qw * qz
        # qxqy, qxqz, qyqz = qx * qy, qx * qz, qy * qz
        # R = np.asarray([[1 - 2 * (qyqy + qzqz),     2 * (qxqy - qwqz),     2 * (qxqz + qwqy)],
        #                 [2 * (qxqy + qwqz),     1 - 2 * (qxqx + qzqz),     2 * (qyqz - qwqx)],
        #                 [2 * (qxqz - qwqy),         2 * (qyqz + qwqx), 1 - 2 * (qxqx + qyqy)]])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_dcm()

        T = np.concatenate((
            np.concatenate((R, t), axis=-1),
            np.asarray([[0, 0, 0, 1]])),
            axis=-2)

        return T

    def restore_image(self, x):
        img = x.transpose((1, 2, 0))
        img = img * self.RGB_STD + self.RGB_MEAN
        img = np.round(img).astype(np.uint8)

        return img


def depth2vis(depth, th=255):
    return np.round(cm.plasma(1 / np.clip(depth + 1, 1, th)) * 255)[:, :, :3].astype(np.uint8)


def create_point_cloud(pt3d, color, zlim=50):
    pt3d = np.reshape(pt3d, (-1, 3))
    color = np.reshape(color, (-1, 3)) / 255

    pcd = o3d.geometry.PointCloud()
    if pt3d.shape == color.shape:
        pcd.points = o3d.utility.Vector3dVector(pt3d[pt3d[:, 2] < zlim])
        pcd.colors = o3d.utility.Vector3dVector(color[pt3d[:, 2] < zlim])
    else:
        print("Invalid input shape.")

    return pcd


def reconstruct_image(image, depth, K):
    h, w = image.shape[:2]

    ys, xs = np.meshgrid(
        np.linspace(0, h - 1, h, dtype=np.float32),
        np.linspace(0, w - 1, w, dtype=np.float32), indexing='ij',
        copy=False)
    ons = np.ones((h, w), dtype=np.float32)

    grid = np.concatenate((xs[None], ys[None], ons[None]))
    grid = np.expand_dims(grid.transpose(1, 2, 0), axis=-1)

    pt3d = np.matmul(inv(K), grid) * depth.reshape(h, w, 1, 1)

    return create_point_cloud(pt3d, image)


def merge_meanvar(mean1, mean2, var1, var2, num1, num2):
    def merge_mean(m1, m2, n1, n2):
        return (m1 * n1 + m2 * n2) / (n1 + n2)

    merged_mean = merge_mean(mean1, mean2, num1, num2)

    mean_square1 = var1 + mean1 ** 2
    mean_square2 = var2 + mean2 ** 2

    merged_mean_square = merge_mean(mean_square1, mean_square2, num1, num2)
    merged_var = merged_mean_square - merged_mean ** 2

    return merged_mean, merged_var


def calc_statistics(images, progress=False):
    mean = None
    var = None
    num = None
    for i, path in enumerate(images):
        x = np.asarray(Image.open(path)).reshape(-1, 3)
        m = np.mean(x, axis=0)
        v = np.var(x, axis=0)
        n = x.shape[0]

        if mean is None:
            mean, var, num = m, v, n
        else:
            mean, var = merge_meanvar(mean, m, var, v, num, n)
            num += n

        if progress:
            print(f'{i+1} / {len(images)}')

    return mean, np.sqrt(var)


def test_merge_meanvar():
    a = np.random.rand(100, 100)
    b = np.random.rand(100, 100)
    c = np.concatenate([a, b], axis=-1)
    expected_mean, expected_var = np.mean(c), np.var(c)

    mean1, var1 = np.mean(a), np.var(a)
    mean2, var2 = np.mean(b), np.var(b)
    actual_mean, actual_var = merge_meanvar(mean1, mean2, var1, var2, len(a), len(b))

    assert np.max(np.abs(actual_mean - expected_mean)) < 1e-8, 'mean error is too large.'
    assert np.max(np.abs(actual_var - expected_var)) < 1e-8, 'var error is toot large.'


def test_calc_statistics(images):
    img_stack = np.stack([Image.open(p) for p in images])
    rgb = img_stack.reshape(-1, 3)
    expected_mean = np.mean(rgb, axis=0)
    expected_std = np.std(rgb, axis=0)

    actual_mean, actual_std = calc_statistics(images)

    assert np.max(np.abs(actual_mean - expected_mean)) < 1e-8, 'mean error is too large.'
    assert np.max(np.abs(actual_std - expected_std)) < 1e-8, 'std error is toot large.'


if __name__ == '__main__':

    print('loading dataset...')
    data_loader = TartanAirLoader(sys.argv[1] if len(sys.argv) > 1 else './TartanAir')
    print()

    while True:
        cin = input('calculate statistics? y/n ')
        if cin == 'y':
            print('testing module... ', end='')
            images = [d['image'] for d in data_loader.dataset[data_loader.testset[0]][:10]]
            test_merge_meanvar()
            test_calc_statistics(images)
            print('OK.')

            print('calculating statistics...')
            images = [d['image'] for t in data_loader.trainset for d in data_loader.dataset[t]]
            mean, std = calc_statistics(images, progress=True)

            np.set_printoptions(precision=16, floatmode='fixed')
            print(f'mean: {mean}')
            print(f'std: {std}')

            data_loader.RGB_MEAN = mean.astype(np.float32)
            data_loader.RGB_STD = std.astype(np.float32)

            break

        elif cin == 'n':
            break
    print()

    text = 'show samples? y/n '
    while True:
        cin = input(text)
        if cin == 'y':
            (x1, x2), (d1, d2), (g1, g2) = data_loader.get_batch(1)
            idx = 0

            image1 = data_loader.restore_image(x1[idx])
            depth1 = depth2vis(d1[idx])
            image2 = data_loader.restore_image(x2[idx])
            depth2 = depth2vis(d2[idx])

            Image.fromarray(np.concatenate(
                (np.concatenate((image1, depth1), axis=1),
                 np.concatenate((image2, depth2), axis=1)),
                axis=0)
            ).show()

            pcd1 = reconstruct_image(image1, d1[idx], data_loader.K)
            cf1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
            pcd2 = reconstruct_image(image2, d2[idx], data_loader.K)
            cf2 = o3d.geometry.TriangleMesh.create_coordinate_frame()
            g = np.matmul(inv(g1[idx]), g2[idx])
            pcd2.transform(g)
            cf2.transform(g)

            o3d.visualization.draw_geometries([pcd1, pcd2, cf1, cf2])

            text = 'again? y/n '

        if cin == 'n':
            break
