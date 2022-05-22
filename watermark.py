#!/usr/bin/env python3
"""主要实现了《基于机器学习和智能优化的数字图像鲁棒盲水印方法研究》这篇文章所述的算法
"""
import itertools
import os
from tabnanny import verbose
import time
import cv2
import functools
import pywt
import numpy as np
import skimage
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sklearn.svm import SVC
import sko
from sko.PSO import PSO
from sko.tools import set_run_mode
import matplotlib.pyplot as plt
from yaml import dump


class SVM_Dataset:
    def __init__(self, dump_file_name='output/embed_pos_svm_dataset.npz'):
        self._toggle_adding_data = True
        self._svm_x = None
        self._svm_y = None
        self._dump_file_name = dump_file_name
        if os.path.exists(self._dump_file_name):
            data = np.load(self._dump_file_name)
            self._svm_x = data['x']
            self._svm_y = data['y']
            print(
                f'Loaded {self._dump_file_name}, {len(self._svm_x)} samples.')

    def add_data(self, x, label):
        if self._svm_x is None:
            self._svm_x = x
            self._svm_y = label
        else:
            self._svm_x = np.concatenate((self._svm_x, x))
            self._svm_y = np.concatenate((self._svm_y, label))

    @property
    def toggle_adding_data(self):
        return self._toggle_adding_data

    @toggle_adding_data.setter
    def toggle_adding_data(self, value):
        self._toggle_adding_data = value

    def dump(self):
        """Dump as npz."""
        np.savez(self._dump_file_name, x=self._svm_x, y=self._svm_y)

    def get_dataset(self):
        return self._svm_x, self._svm_y


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def compare_nc(img1, img2):
    return np.mean(np.multiply(img1 - np.mean(img1), img2 - np.mean(img2))) / np.std(img1) / np.std(img2)


def attack_gaussian_noise(image, var=0.001):
    return skimage.img_as_ubyte(skimage.util.random_noise(image, mode='gaussian', var=var))


def attack_salt(image, amount):
    return skimage.img_as_ubyte(skimage.util.random_noise(image, mode='salt', amount=amount))


def attack_pepper(image, amount):
    return skimage.img_as_ubyte(skimage.util.random_noise(image, mode='pepper', amount=amount))


def attack_salt_and_pepper(image, amount):
    return skimage.img_as_ubyte(skimage.util.random_noise(image, mode='s&p', amount=amount))


def attack_speckle(image, mean, var):
    return skimage.img_as_ubyte(skimage.util.random_noise(image, mode='speckle', mean=mean, var=var))


def arnold_transform(img, key=5):
    r = img.shape[0]
    c = img.shape[1]
    p = np.zeros((r, c), np.uint8)
    a = 1
    b = 1
    for k in range(key):
        for i in range(r):
            for j in range(c):
                x = (i + b * j) % r
                y = (a * i + (a * b + 1) * j) % c
                p[x, y] = img[i, j]
    return p


def arnold_invert_transform(img, key=5):
    r = img.shape[0]
    c = img.shape[1]
    p = np.zeros((r, c), np.uint8)
    a = 1
    b = 1
    for k in range(key):
        for i in range(r):
            for j in range(c):
                x = ((a * b + 1) * i - b * j) % r
                y = (-a * i + j) % c
                p[x, y] = img[i, j]
    return p


def convert_bgr_to_ycc(image):
    """
    Convert an image from BGR to YCC.
    """
    ycc_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    return ycc_image


def convert_ycc_to_bgr(image):
    """
    Convert an image from YCC to BGR.
    """
    bgr_image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
    return bgr_image


def dwt(image):
    """
    Perform a discrete wavelet transform on an image.
    """
    return pywt.wavedec2(image, 'haar', level=1)


def idwt(image):
    return pywt.waverec2(image, 'haar')


def dct(image):
    return cv2.dct(image)


def idct(image):
    return cv2.idct(image)


class Partitions(object):
    def __init__(self, partitions, partition_size, height, width):
        self.partitions = partitions
        self.partition_size = partition_size
        self.height = height
        self.width = width

    def __iter__(self):
        return iter(self.partitions)

    def recombine(self):
        N = self.partition_size
        combined_image = np.zeros(
            (self.height * N, self.width * N), self.partitions[0].dtype)
        for i in range(self.height):
            for j in range(self.width):
                combined_image[i * N:(i + 1) * N, j *
                               N:(j + 1) * N] = self.partitions[i*self.width + j]
        return combined_image

    @classmethod
    def from_image(cls, image, partition_size):
        # image coordinates: (y, x, channel)
        height = image.shape[0] // partition_size
        width = image.shape[1] // partition_size
        partitions = []
        for i in range(height):
            for j in range(width):
                partitions.append(image[i * partition_size:(i + 1) *
                                        partition_size, j * partition_size:(j + 1) * partition_size])
        return cls(partitions, partition_size, height, width)

    @classmethod
    def from_list(cls, lst, ps):
        return cls(lst, ps.partition_size, ps.height, ps.width)


def z_scanning_image_index(i):
    mapping = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1),
               (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4)]
    return mapping[i]


def extract_modulation_matrix(src_mat):
    M = np.empty(16, src_mat.dtype)
    for i in range(16):
        src_ind_y, src_ind_x = z_scanning_image_index(i)
        M[i] = src_mat[src_ind_y, src_ind_x]
    return M.reshape(4, 4)


def apply_modulation_matrix(dst_mat, M):
    ret = dst_mat.copy()
    for i in range(16):
        dst_ind_y, dst_ind_x = z_scanning_image_index(i)
        ret[dst_ind_y, dst_ind_x] = M[i//4, i % 4]
    return ret


def embed_watermark_in_svds(watermark, SVDs, embed_model, a):
    height = watermark.shape[0]
    width = watermark.shape[1]

    # model 1: sort the SVDs by the first eigenvalue (descending), and embed the watermark in
    # the first N SVDs, where N is the size of the watermark
    if False:
        sigma1 = np.array([svd[1][0] for svd in SVDs])
        indices = np.argsort(sigma1, axis=None)

        if indices.shape[0] < height*width:
            raise ValueError(
                'Watermark image is too large to embed in the image.')

        embed_indices = indices[:height*width]

        # save model 1
        if g_dataset.toggle_adding_data:
            g_dataset.add_data(np.array([svd[1] for svd in SVDs])[indices], np.array(
                [1]*len(embed_indices)+[0]*(len(indices)-len(embed_indices))))

    # model 2: SVM
    if False:
        eigenvalues = np.array([svd[1] for svd in SVDs])
        label = embed_model.predict(eigenvalues)
        embed_indices, = np.nonzero(label)
        embed_indices = embed_indices[:height*width]

    # model 3: embed the watermark in the first N SVDs, where N is the size of the watermark
    if True:
        embed_indices = range(height*width)

    # embed watermark
    for i, ind in enumerate(embed_indices):
        u, s, vh = SVDs[ind]
        if watermark[i//width, i % width] == 255:
            s[1] = a * s[0] + (1 - a) * s[1]
        else:
            s[1] = (1 - a) * s[1] + a * s[2]
    return embed_indices


def extract_watermark_from_svds(SVDs, embed_indices, extract_model, watermark_shape):
    """
    Args:
        SVDs (): 
        embed_indices (list[int]): indices of the SVDs that contain the watermark
        watermark_shape (tuple[int]): shape of the watermark returned

    Returns:
        img: watermark image
    """

    watermark = np.zeros(len(embed_indices), dtype=np.uint8)
    if extract_model is None:
        for i, ind in enumerate(embed_indices):
            u, s, vh = SVDs[ind]
            if s[0] - s[1] <= s[1] - s[2]:
                watermark[i] = 255
            else:
                watermark[i] = 0
    else:
        watermark = (extract_model.predict(
            np.array([SVDs[ind][1] for ind in embed_indices])) * 255).astype('uint8')
    return watermark.reshape(watermark_shape)


def embed_watermark(img, watermark, embed_model, a=0.7):
    img_ycc = convert_bgr_to_ycc(img)

    img_y = img_ycc[:, :, 0]

    # DWT
    [cA, (cH, cV, cD)] = dwt(img_y)

    cA_8n = cA[:cA.shape[0]//8*8, :cA.shape[1]//8*8]  # 8n x 8n slice of cA

    partitions = Partitions.from_image(cA_8n, 8)

    # DCT
    DCTs = list(map(dct, partitions))

    mod_matrices = map(extract_modulation_matrix, DCTs)

    # SVD
    SVDs = list(map(lambda x: np.linalg.svd(x, compute_uv=True), mod_matrices))

    watermark_ = arnold_transform(watermark)

    # Embed watermark
    embed_indices = embed_watermark_in_svds(watermark_, SVDs, embed_model, a=a)

    # ISVD
    mod_matrices_embedded = itertools.starmap(
        lambda u, s, vh: u @ np.diag(s) @ vh, SVDs)

    # Assign modulated value to DCTs
    DCTs_embedded = list(map(apply_modulation_matrix,
                             DCTs, mod_matrices_embedded))

    # IDCT
    partitions_embed = Partitions.from_list(
        list(map(idct, DCTs_embedded)), partitions)

    cA_8n_embedded = partitions_embed.recombine()
    cA[:cA.shape[0]//8*8, :cA.shape[1]//8*8] = cA_8n_embedded

    # IDWT
    img_y = idwt([cA, (cH, cV, cD)])
    img_ycc[:, :, 0] = img_y.astype(np.uint8)
    img_bgr = convert_ycc_to_bgr(img_ycc)
    return img_bgr, embed_indices


def extract_watermark(img, embed_indices, extract_model, watermark_shape):
    img_ycc = convert_bgr_to_ycc(img)
    img_y = img_ycc[:, :, 0]
    # DWT
    [cA, (cH, cV, cD)] = dwt(img_y)

    cA_8n = cA[:cA.shape[0]//8*8, :cA.shape[1]//8*8]  # 8n x 8n slice of cA

    # DCT
    DCTs = map(dct, Partitions.from_image(cA_8n, 8))

    modulation_matrices = map(extract_modulation_matrix, DCTs)

    # SVD
    SVDs = list(map(lambda x: np.linalg.svd(
        x, compute_uv=True), modulation_matrices))

    watermark = extract_watermark_from_svds(
        SVDs, embed_indices, extract_model, watermark_shape)
    watermark = arnold_invert_transform(watermark)
    return watermark, SVDs


def generate_optimization_problem(img, watermark, embed_model, extract_model, verbose=False):
    lam = 42

    def fitness(a):
        watermarked_img, embed_indices = embed_watermark(
            img, watermark, embed_model, a=a)

        attacked_img = attack_gaussian_noise(watermarked_img, 0.001)

        extracted_watermark, _ = extract_watermark(
            attacked_img, embed_indices, extract_model, watermark.shape)
        # plt.imshow(extracted_watermark)

        psnr = compare_psnr(img, watermarked_img)
        nc = compare_nc(watermark, extracted_watermark)
        if verbose:
            print(f'a={a}, psnr={psnr}, nc={nc}')

        return - psnr - lam * nc
    return fitness


def test_with(func):
    imgs = dict((f, cv2.imread(f))
                for f in ['./test/BaboonRGB.bmp', './test/LenaRGB.bmp', './test/PeppersRGB.bmp', './test/1.bmp'])
    watermarks = dict((f, cv2.imread(f, 0))
                      for f in ['./test/watermark1.png', './test/watermark2.png'])

    def attack_gaussian_noise_0_001(img):
        return attack_gaussian_noise(img, var=0.001)

    def attack_gaussian_noise_0_005(img):
        return attack_gaussian_noise(img, var=0.005)

    def attack_salt_and_pepper_0_001(img):
        return attack_salt_and_pepper(img, amount=0.001)

    def attack_speckle_0_001(img):
        return attack_speckle(img, mean=0, var=0.001)

    attacks = [
        attack_gaussian_noise_0_001,
        attack_gaussian_noise_0_005,
        attack_salt_and_pepper_0_001,
        attack_speckle_0_001
    ]
    for (img_path, watermark_path, attack) in itertools.product(imgs.keys(), watermarks.keys(), attacks):
        print(
            f'test with: img_path={img_path},\t watermark_path={watermark_path},\t attack={attack.__name__}')
        func(imgs[img_path], watermarks[watermark_path], attack)


def create_extract_model():
    extract_dataset = SVM_Dataset(
        dump_file_name='output/extract_svm_dataset.npz')
    extract_dataset.toggle_adding_data = True

    def func(img, watermark, attack):
        watermarked_img, embed_indices = embed_watermark(
            img, watermark, None, a=0.5)

        attacked_img = attack(watermarked_img)

        extracted_watermark, SVDs = extract_watermark(
            attacked_img, embed_indices, None, watermark.shape)

        # plt.figure()
        # plt.imshow(watermarked_img)
        watermark_ = arnold_transform(watermark).flatten()
        for i, ind in enumerate(embed_indices):
            extract_dataset.add_data(
                np.array([SVDs[ind][1]]), np.array([1 if watermark_[i] == 255 else 0]))

    test_with(func)
    extract_dataset.dump()


# create_extract_model()
# plt.show()
# exit(0)

# ============================== Models ================================
start = time.time()
g_dataset = SVM_Dataset()
g_dataset.toggle_adding_data = False

embed_model = SVC(kernel='linear')
embed_model.fit(*g_dataset.get_dataset())

extract_dataset = SVM_Dataset(dump_file_name='output/extract_svm_dataset.npz')
extract_model = SVC()
extract_model.fit(*extract_dataset.get_dataset())
print(f'Training models time: {time.time() - start}')

# ======================= embed and extract ============================
if False:
    # read image
    img = cv2.imread('./9.bmp')
    watermark = cv2.imread('./watermark.png', 0)

    assert((arnold_invert_transform(arnold_transform(
        watermark)) == watermark).all())

    print(f'Shape of source image: {img.shape}')
    print(f'Shape of watermark image: {watermark.shape}')

    start = time.time()
    watermarked, embed_indices = embed_watermark(img, watermark, embed_model)
    extracted_watermark, _ = extract_watermark(
        watermarked, embed_indices, None, watermark.shape)
    print(f'Running embed/extract time: {time.time() - start}')

    cv2.imwrite('./output/watermarked.png', watermarked)
    cv2.imwrite('./output/extracted_watermark.png', extracted_watermark)

    # calculate psnr
    psnr = compare_psnr(img, watermarked)
    nc = compare_nc(watermark, extracted_watermark)
    print(f'PSNR: {psnr}, NC: {nc}')

    if g_dataset.toggle_adding_data:
        g_dataset.dump()

    exit(0)

# ========================= test optimization problem ============================


def test_svm_extract():
    for img_path in ['LenaRGB.bmp', 'BaboonRGB.bmp', 'PeppersRGB.bmp']:
        img = cv2.imread(f'./test/{img_path}')
        print('Testing with image:', img_path)
        watermark = cv2.imread('./test/watermark1.png', 0)
        fitness = generate_optimization_problem(
            img, watermark, None, extract_model, verbose=True)
        fitness2 = generate_optimization_problem(
            img, watermark, None, None, verbose=True)

        print('Using SVM to extract watermark')
        fitness(0.5)
        print('Using normal algorithm to extract watermark')
        fitness2(0.5)
    plt.show()


test_svm_extract()
exit(0)


def fitness_wrapper(a):
    return fitness(a)


# set_run_mode require fitness to be a function, thus need to wrap it
set_run_mode(fitness_wrapper, 'multiprocessing')

fitness_v = generate_optimization_problem(
    img, watermark, embed_model, verbose=True)

start = time.time()
pso = PSO(func=fitness_wrapper, n_dim=1, pop=5, max_iter=50,
          lb=[0.2], ub=[1], w=0.2, c1=0.5, c2=0.5)
pso.run()
print(f'Running PSO time: {time.time() - start}')

print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
fitness_v(pso.gbest_x)

plt.plot(pso.gbest_y_hist)
plt.show()
