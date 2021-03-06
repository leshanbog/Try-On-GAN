import torchvision
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model):
    device = next(model.parameters()).device

    classifier = torchvision.models.mobilenet_v2(pretrained=True).to(device)

    classifier.eval()
    model.eval()

    m1, s1, m2, s2 = None, None, None, None

    activations_size = 5120

    p1 = np.empty(((len(dataloader) - 1) * dataloader.batch_size, activations_size))
    p2 = np.empty_like(p1)

    index = 0
    for img1, mask1, cc1, cc2 in tqdm(dataloader, leave=False, total=len(dataloader)):
        img1 = img1.to(device)
        cc2 = cc2.to(device)
        mask1 = mask1.to(device)
        bs = img1.shape[0]

        out = model(img1, cc2, mask1)['out']

        assert out.shape == img1.shape

        a1 = classifier.features(img1).detach().cpu().numpy().reshape(bs, -1)
        a2 = classifier.features(out).detach().cpu().numpy().reshape(bs, -1)

        p1[index: index + bs] = a1[:, :activations_size]
        p2[index: index + bs] = a2[:, :activations_size]

        index += dataloader.batch_size

        if index >= len(p1):
            break

    return p1.mean(0), np.cov(p1, rowvar=False), p2.mean(0), np.cov(p2, rowvar=False)


@torch.no_grad()
def calculate_fid(dataloader, generator):

    m1, s1, m2, s2 = calculate_activation_statistics(dataloader, generator)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()