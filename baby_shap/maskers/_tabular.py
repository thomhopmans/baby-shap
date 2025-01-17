import logging

import numpy as np
import pandas as pd

from baby_shap.maskers._masker import Masker
from baby_shap.utils._clustering import hclust
from baby_shap.utils._exceptions import DimensionError, InvalidClusteringError
from baby_shap.utils._general import safe_isinstance, sample

log = logging.getLogger("baby_shap")


class Tabular(Masker):
    """A common base class for Independent and Partition."""

    def __init__(self, data, max_samples=100, clustering=None):
        """This masks out tabular features by integrating over the given background dataset.

        Parameters
        ----------
        data : np.array, pandas.DataFrame
            The background dataset that is used for masking.

        max_samples : int
            The maximum number of samples to use from the passed background data. If data has more
            than max_samples then shap.utils.sample is used to subsample the dataset. The number of
            samples coming out of the masker (to be integrated over) matches the number of samples in
            the background dataset. This means larger background dataset cause longer runtimes. Normally
            about 1, 10, 100, or 1000 background samples are reasonable choices.

        clustering : string or None (default) or numpy.ndarray
            The distance metric to use for creating the clustering of the features. The
            distance function can be any valid scipy.spatial.distance.pdist's metric argument.
            However we suggest using 'correlation' in most cases. The full list of options is
            ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
            ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
            ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
            ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’. These are all
            the options from scipy.spatial.distance.pdist's metric argument.
        """

        self.output_dataframe = False
        if safe_isinstance(data, "pandas.core.frame.DataFrame"):
            self.feature_names = data.columns
            data = data.values
            self.output_dataframe = True

        if isinstance(data, dict) and "mean" in data:
            self.mean = data.get("mean", None)
            self.cov = data.get("cov", None)
            data = np.expand_dims(data["mean"], 0)

        if hasattr(data, "shape") and data.shape[0] > max_samples:
            data = sample(data, max_samples)

        self.data = data
        self.clustering = clustering
        self.max_samples = max_samples

        # compute the clustering of the data
        if clustering is not None:
            if isinstance(clustering, str):
                self.clustering = hclust(data, metric=clustering)
            elif safe_isinstance(clustering, "numpy.ndarray"):
                self.clustering = clustering
            else:
                raise InvalidClusteringError(
                    "Unknown clustering given! Make sure you pass a distance metric as a string, "
                    "or a clustering as a numpy.ndarray."
                )
        else:
            self.clustering = None

        self._masked_data = data.copy()
        self._last_mask = np.zeros(data.shape[1], dtype=bool)
        self.shape = self.data.shape
        self.supports_delta_masking = True

    def __call__(self, mask, x):
        mask = self._standardize_mask(mask, x)

        # make sure we are given a single sample
        if len(x.shape) != 1 or x.shape[0] != self.data.shape[1]:
            raise DimensionError(
                "The input passed for tabular masking does not match the background data shape!"
            )

        # if mask is an array of integers then we are doing delta masking
        if np.issubdtype(mask.dtype, np.integer):

            variants = ~self.invariants(x)
            curr_delta_inds = np.zeros(len(mask), dtype=np.int)
            num_masks = (mask >= 0).sum()
            varying_rows_out = np.zeros((num_masks, self.shape[0]), dtype=np.bool)
            masked_inputs_out = np.zeros((num_masks * self.shape[0], self.shape[1]))
            self._last_mask[:] = False
            self._masked_data[:] = self.data

            delta_mask_noop_value = (
                2147483647  # used to encode a noop for delta masking
            )
            _delta_masking(
                mask,
                x,
                curr_delta_inds,
                varying_rows_out,
                self._masked_data,
                self._last_mask,
                self.data,
                variants,
                masked_inputs_out,
                delta_mask_noop_value,
            )
            if self.output_dataframe:
                return (
                    pd.DataFrame(masked_inputs_out, columns=self.feature_names),
                ), varying_rows_out

            return (masked_inputs_out,), varying_rows_out

        # otherwise we update the whole set of masked data for a single sample
        self._masked_data[:] = x * mask + self.data * np.invert(mask)
        self._last_mask[:] = mask

        if self.output_dataframe:
            return pd.DataFrame(self._masked_data, columns=self.feature_names)

        return (self._masked_data,)

    def invariants(self, x):
        """This returns a mask of which features change when we mask them.

        This optional masking method allows explainers to avoid re-evaluating the model when
        the features that would have been masked are all invariant.
        """

        # make sure we got valid data
        if x.shape != self.data.shape[1:]:
            raise DimensionError(
                "The passed data does not match the background shape expected by the masker! The data of shape "
                + str(x.shape)
                + " was passed while the masker expected data of shape "
                + str(self.data.shape[1:])
                + "."
            )

        return np.isclose(x, self.data)


def _single_delta_mask(dind, masked_inputs, last_mask, data, x, noop_code):
    if dind == noop_code:
        pass
    elif last_mask[dind]:
        masked_inputs[:, dind] = data[:, dind]
        last_mask[dind] = False
    else:
        masked_inputs[:, dind] = x[dind]
        last_mask[dind] = True


def _delta_masking(
    masks,
    x,
    curr_delta_inds,
    varying_rows_out,
    masked_inputs_tmp,
    last_mask,
    data,
    variants,
    masked_inputs_out,
    noop_code,
):
    """Implements the special (high speed) delta masking API that only flips the positions we need to.

    Note that we attempt to avoid doing any allocation inside this function for speed reasons.
    """

    dpos = 0
    i = -1
    masks_pos = 0
    output_pos = 0
    N = masked_inputs_tmp.shape[0]
    while masks_pos < len(masks):
        i += 1

        # update the tmp masked inputs array
        dpos = 0
        curr_delta_inds[0] = masks[masks_pos]
        while curr_delta_inds[dpos] < 0:  # negative values mean keep going
            curr_delta_inds[dpos] = (
                -curr_delta_inds[dpos] - 1
            )  # -value + 1 is the original index that needs flipped
            _single_delta_mask(
                curr_delta_inds[dpos], masked_inputs_tmp, last_mask, data, x, noop_code
            )
            dpos += 1
            curr_delta_inds[dpos] = masks[masks_pos + dpos]
        _single_delta_mask(
            curr_delta_inds[dpos], masked_inputs_tmp, last_mask, data, x, noop_code
        )

        # copy the tmp masked inputs array to the output
        masked_inputs_out[output_pos : output_pos + N] = masked_inputs_tmp
        masks_pos += dpos + 1

        # mark which rows have been updated, so we can only evaluate the model on the rows we need to
        if i == 0:
            varying_rows_out[i, :] = True

        else:
            # only one column was changed
            if dpos == 0:
                varying_rows_out[i, :] = variants[:, curr_delta_inds[dpos]]

            # more than one column was changed
            else:
                varying_rows_out[i, :] = (
                    np.sum(variants[:, curr_delta_inds[: dpos + 1]], axis=1) > 0
                )

        output_pos += N


class Independent(Tabular):
    """This masks out tabular features by integrating over the given background dataset."""

    def __init__(self, data, max_samples=100):
        """Build a Independent masker with the given background data.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame
            The background dataset that is used for masking.

        max_samples : int
            The maximum number of samples to use from the passed background data. If data has more
            than max_samples then shap.utils.sample is used to subsample the dataset. The number of
            samples coming out of the masker (to be integrated over) matches the number of samples in
            the background dataset. This means larger background dataset cause longer runtimes. Normally
            about 1, 10, 100, or 1000 background samples are reasonable choices.
        """
        super().__init__(data, max_samples=max_samples, clustering=None)


class Partition(Tabular):
    """This masks out tabular features by integrating over the given background dataset.

    Unlike Independent, Partition respects a hierarchial structure of the data.
    """

    def __init__(self, data, max_samples=100, clustering="correlation"):
        """Build a Partition masker with the given background data and clustering.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame
            The background dataset that is used for masking.

        max_samples : int
            The maximum number of samples to use from the passed background data. If data has more
            than max_samples then shap.utils.sample is used to subsample the dataset. The number of
            samples coming out of the masker (to be integrated over) matches the number of samples in
            the background dataset. This means larger background dataset cause longer runtimes. Normally
            about 1, 10, 100, or 1000 background samples are reasonable choices.

        clustering : string or numpy.ndarray
            If a string, then this is the distance metric to use for creating the clustering of
            the features. The distance function can be any valid scipy.spatial.distance.pdist's metric
            argument. However we suggest using 'correlation' in most cases. The full list of options is
            ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
            ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
            ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
            ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’. These are all
            the options from scipy.spatial.distance.pdist's metric argument.
            If an array, then this is assumed to be the clustering of the features.
        """
        super().__init__(data, max_samples=max_samples, clustering=clustering)


class Impute(
    Masker
):  # we should inherit from Tabular once we add support for arbitrary masking
    """This imputes the values of missing features using the values of the observed features.

    Unlike Independent, Gaussian imputes missing values based on correlations with observed data points.
    """

    def __init__(self, data, method="linear"):
        """Build a Partition masker with the given background data and clustering.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame or {"mean: numpy.ndarray, "cov": numpy.ndarray} dictionary
            The background dataset that is used for masking.
        """
        if data is dict and "mean" in data:
            self.mean = data.get("mean", None)
            self.cov = data.get("cov", None)
            data = np.expand_dims(data["mean"], 0)

        self.data = data
        self.method = method
