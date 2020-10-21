"""
create by: chenjianyi
create time: 2020.10.09 11:36
references: https://github.com/MarcoForte/bayesian-matting
            Yung-Yu Chuang, Brian Curless, David H. Salesin, and Richard Szeliski. A Bayesian Approach to Digital Matting. In Proceedings of IEEE Computer Vision and Pattern Recognition (CVPR 2001), Vol. II, 264-271, December 2001
"""
import numpy as np
import cv2

from orchard_bouman_clust import clustFunc

class BayesianMatting():
    """A Bayesian Approach to Digital Matting.
    Params:
        sigma: used for estimating L(F), `we use a spatial Gaussian falloff gi with σ=8 to stress the contribution of nearby pixels 
               over those that are further away`
        N: used for estimating L(F), `we weight the contribution of each nearby pixel i in N accord- ing to two separate factors`
        minN: 
    """
    def __init__(self, sigma=8, N=25, minN=10):
        self.sigma = sigma
        self.N = N
        self.minN = minN

    def gaussian2d(self, shape=(3, 3), sigma=0.5):
        """same as MATLAB's fspecial('gaussian',[shape],[sigma])
        Returns:
            h: (shape[0], shape[1]) <==> (m, n)
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m: m+1, -n: n+1]
        h = np.exp(-(x * x + y * y)) / (2. * sigma * sigma)
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_window(self, m, x, y, N):
        h, w, c = m.shape
        halfN = N // 2
        r = np.zeros((N, N, c))
        xmin = max(0, x - halfN)
        xmax = min(w, x + (halfN + 1))
        ymin = max(0, y - halfN)
        ymax = min(h, y + (halfN + 1))
        pxmin = halfN - (x - xmin)
        pxmax = halfN + (xmax - x)
        pymin = halfN - (y - ymin)
        pymax = halfN + (ymax - y)
        r[pymin: pymax, pxmin: pxmax] = m[ymin: ymax, xmin: xmax]
        return r

    def solve(self, mu_f, sigma_f, mu_b, sigma_b, c, sigma_c, alpha_init, max_iter, min_likelihood):
        """Solves for F,B and alpha that maximize the sum of log likelihoods at the given pixel C.
        Params:
            mu_f:           means of foreground clusters, (Fclusters, 3)
            sigma_f:        covariances of foreground clusters, (Fclusters, 3, 3)
            mu_b:           means of background clusters, (Bclusters, 3)
            sigma_f:        covariances of background clusters, (Bclusters, 3, 3)
            c:              observed pixel, (3, )
            sigma_c:        covariances of c
            alpha_init:     initial alpha value, (1, )
            max_iter:       maximal number of iterations
            min_likelihood: minimal change in likelihood between consecutive iterations
        Returns:
            FMax:     F value, (3, )
            BMax:     B value, (3, )
            alphaMax: alpha value, (1, )
        """
        I = np.eye(3)
        FMax = np.zeros(3)
        BMax = np.zeros(3)
        alphaMax = 0
        maxLike = -np.inf
        invsigma2 = 1 / sigma_c ** 2
        for i in range(mu_f.shape[0]):
            mu_fi = mu_f[i] #(3, )
            invsigma_fi = np.linalg.inv(sigma_f[i])  #(3, 3)
            for j in range(mu_b.shape[0]):
                mu_bj = mu_b[j] #(3, )
                invsigma_bj = np.linalg.inv(sigma_b[j]) # (3, 3)
                alpha = alpha_init
                curr_iter = 1
                lastLike = -1.7977e308
                while(1):
                    # solve for F, B, fixing alpha
                    A11 = invsigma_fi + I * alpha**2 * invsigma2 #(3, 3)
                    A12 = I * alpha * (1 - alpha) * invsigma2  #(3, 3)
                    A22 = invsigma_bj + I * (1 - alpha)**2 * invsigma2 #(3, 3)
                    A = np.vstack((np.hstack((A11, A12)), np.hstack((A12, A22))))  # (6, 6)
                    b1 = invsigma_fi @ mu_fi + c * alpha * invsigma2  # (3, )
                    b2 = invsigma_bj @ mu_bj + c * (1 - alpha) * invsigma2 #(3, )
                    b = np.atleast_2d(np.concatenate((b1, b2))).T  #(6, 1)
                    X = np.linalg.solve(A, b)  # (6, 1)
                    F = np.maximum(0, np.minimum(1, X[0: 3])) #(3, 1)
                    B = np.maximum(0, np.minimum(1, X[3: 6])) #(3, 1)

                    # solve for alpha
                    alpha = (np.atleast_2d(c).T - B).T @ (F - B) / np.sum((F - B) ** 2)  # (1, 1)
                    alpha = np.maximum(0, np.minimum(1, alpha))[0, 0]  #(1, )

                    # calculate likelihood
                    L_C = -np.sum((np.atleast_2d(c).T - alpha * F - (1 - alpha) * B) ** 2) * invsigma2
                    L_F = (-((F - np.atleast_2d(mu_fi).T).T @ invsigma_fi @ (F - np.atleast_2d(mu_fi).T)) / 2)[0, 0]  #(1, 3) @ (3, 3) @ (3, 1)
                    L_B = (-((B - np.atleast_2d(mu_bj).T).T @ invsigma_fi @ (B - np.atleast_2d(mu_bj).T)) / 2)[0, 0]  #(1, 3) @ (3, 3) @ (3, 1)
                    likelihood = L_C + L_F + L_B

                    # early stop
                    if likelihood > maxLike:
                        alphaMax = alpha
                        maxLike = likelihood
                        FMax = F.ravel()
                        BMax = B.ravel()
                    if curr_iter >= max_iter or abs(likelihood - lastLike) <= min_likelihood:
                        break
                    lastLike = likelihood
                    curr_iter += 1

        return FMax, BMax, alphaMax

    def bayesian_matte(self, img, trimap):
        img = img.astype('float') / 255.
        h, w, c = img.shape

        alpha = np.zeros((h, w))  # initial alpha

        # forefround, background, unkown, mask
        fg_mask = (trimap == 255)
        bg_mask = (trimap == 0)
        unknown_mask = True ^ np.logical_or(fg_mask, bg_mask)
        foreground = img * np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2)
        background = img * np.repeat(bg_mask[:, :, np.newaxis], 3, axis=2)

        gaussian_weights = self.gaussian2d((self.N, self.N), self.sigma)
        gaussian_weights = gaussian_weights / np.max(gaussian_weights)

        alpha[fg_mask] = 1
        F = np.zeros(img.shape)
        B = np.zeros(img.shape)
        alphaRes = np.zeros(trimap.shape)

        n = 1
        alpha[unknown_mask] = np.nan
        num_unknown = np.sum(unknown_mask)
        unkreg = unknown_mask.copy()
        kernel = np.ones((3, 3))
        while n < num_unknown:
            unkreg = cv2.erode(unkreg.astype(np.uint8), kernel, iterations=1)
            unkpixels = np.logical_and(np.logical_not(unkreg), unknown_mask)

            Y, X = np.nonzero(unkpixels)
            for i in range(Y.shape[0]):
                y, x = Y[i], X[i]
                p = img[y, x]
                a = self.get_window(alpha[:, :, np.newaxis], x, y, self.N)[:, :, 0]  # surrounding alpha

                ## foreground
                f_pixels = self.get_window(foreground, x, y, self.N)  # surronding foreground pixels
                f_weights = (a**2 * gaussian_weights).ravel()
                f_pixels = np.reshape(f_pixels, (self.N * self.N, 3))
                posInds = np.nan_to_num(f_weights) > 0  # select known alpha
                f_pixels = f_pixels[posInds, :]
                f_weights = f_weights[posInds]

                ## background
                b_pixels = self.get_window(background, x, y, self.N)
                b_weights = ((1-a)**2 * gaussian_weights).ravel()
                b_pixels = np.reshape(b_pixels, (self.N * self.N, 3))
                posInds = np.nan_to_num(b_weights) > 0
                b_pixels = b_pixels[posInds, :]
                b_weights = b_weights[posInds]

                ## if not enough data, return to it later
                if len(f_weights) < self.minN or len(b_weights) < self.minN:
                    continue

                ## cluster
                mu_f, sigma_f = clustFunc(f_pixels, f_weights) #(num_cluster1, 3), (num_cluster1, 3, 3)
                mu_b, sigma_b = clustFunc(b_pixels, b_weights) #(num_cluster2, 3), (num_cluster2, 3, 3)

                alpha_init = np.nanmean(a.ravel())

                ## solver
                f, b, alphaT = self.solve(mu_f, sigma_f, mu_b, sigma_b, p, 0.01, alpha_init, 50, 1e-6)
