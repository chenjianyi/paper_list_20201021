"""
create by: chenjianyi
create time: 2020.10.06 15:14
reference: https://github.com/joyeecheung/dark-channel-prior-dehazing
           Tang X . Single image haze removal using dark channel prior[C]// 2009 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2009.
"""
import cv2
import numpy as np

class HazeRemovalUsingDarkChannelPrior():
    """HazeRemovalUsingDarkChannelPrior
    Params:
        window_size: window size used for estimating dark channel and transmission
        omega:       used for optionally keeping a very small amount of haze, in estimating transmission phase
        p:           used in estimating atmosphere phase, "first pick the top `p` brightest pixels in the dark channel"
        Amax:        A(atmosphere) is not bigger than A max, used in estimating radiance phase
        tmin:        t(transmission) is not smallel than tmin, used in estimating radiance phase
        using_guildedfilter: not implemented
    """
    def __init__(self, window_size=15, omega=0.95, p=0.001, Amax=220, tmin=0.1, using_guildedfilter=False):
        self.window_size = window_size
        self.omega = omega
        self.p = p
        self.Amax = Amax
        self.tmin = tmin
        self.using_guildedfilter = using_guildedfilter

    def get_dark_channel(self, img):
        """get dark channel prior
        Params:
            img: input image, (h, w, c)
        Returns:
            dark_channel: (h, w)
        """
        h, w, c = img.shape
        img_padded = np.pad(img, ((int(self.window_size / 2), int(self.window_size / 2)), (int(self.window_size / 2), int(self.window_size / 2)), (0, 0)), 'edge')
        dark_channel = np.zeros((h, w))
        for i, j in np.ndindex(dark_channel.shape):
            dark_channel[i, j] = np.min(img_padded[i: i+self.window_size, j: j+self.window_size, :])  ##cvpr2009 eq. 5
        return dark_channel

    def get_transmission(self, img, A):
        """estimate transmission
        Params:
            img: image, (h, w, c)
            A:   atmospheric light, (3, )
        Returns:
            transmission: (h, w)
        """
        transmission = 1 - self.omega * self.get_dark_channel(img / A)  ##cvpr2009, eq.12
        return transmission

    def get_atmosphere(self, img, dark_channel):
        h, w = dark_channel.shape
        flat_img = img.reshape(h * w, 3)  # (h*w, 3)
        flat_dark = dark_channel.ravel()  # (h*w, )
        search_idx = (-flat_dark).argsort()[: int(h*w*self.p)]  # find top M * N * p indexes
        A = np.max(flat_img.take(search_idx, axis=0), axis=0)  #(3,)
        return A

    def get_radiance(self, img, A, t):
        tiled_t = np.zeros_like(img)
        tiled_t[:, :, 0] = tiled_t[:, :, 1] = tiled_t[:, :, 2] = t
        radiance = (img - A) / tiled_t + A
        return radiance # (h, w, 3)

    def dehaze(self, img_):
        img = np.asarray(img_, dtype=np.float64)
        h, w, _ = img.shape

        dark = self.get_dark_channel(img)  # get dark channel

        A = self.get_atmosphere(img, dark) # get raw atmosphere
        A = np.minimum(A, self.Amax)  # A should be not bigger than Amax

        t = self.get_transmission(img, A)  # get raw transmission
        t = np.maximum(t, self.tmin)  # t should be not smaller than tmin

        radiance = self.get_radiance(img, A, t)  # get radiance

        return self._to_image(dark), self._to_image(radiance), A, t

    def _to_image(self, img):
        img = np.maximum(np.minimum(img, 255), 0).astype(np.uint8)
        return img
