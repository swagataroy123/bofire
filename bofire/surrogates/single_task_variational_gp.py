import base64
import io
from typing import Dict, Optional

import botorch
import dill
import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import StudentTLikelihood
from gpytorch.mlls import VariationalELBO

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import (
    SingleTaskVariationalGPSurrogate as DataModel,
)
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.single_task_gp import get_scaler
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs


class SingleTaskVariationalGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.scaler = data_model.scaler
        self.noise_prior = data_model.noise_prior
        self.num_outputs = data_model.num_outputs
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.SingleTaskVariationalGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)
        self.output_transform = Standardize(m=tY.shape[-1])
        tY, _ = self.output_transform(tY)
        self.model = botorch.models.SingleTaskVariationalGP(  # type: ignore
            train_X=tX,
            train_Y=tY,
            likelihood=StudentTLikelihood(noise_prior=priors.map(self.noise_prior)),
            num_outputs=self.num_outputs,
            learn_inducing_points=False,
            inducing_points=tX,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(tX.shape[1])),
                ard_num_dims=1,  # this keyword is ingored
            ),
            # outcome_transform=Standardize(m=tY.shape[-1]),
            input_transform=scaler,
        )

        # self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)  # type: ignore

        mll = VariationalELBO(
            self.model.likelihood, self.model.model, num_data=tX.shape[-2]
        )
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        self.model.model.eval()  # type: ignore
        self.model.likelihood.eval()  # type: ignore
        with torch.no_grad() and gpytorch.settings.num_likelihood_samples(128):
            preds = self.model.posterior(X=X, observation_noise=True).mean.mean(dim=0).cpu().detach()  # type: ignore
            variance = self.model.posterior(X=X, observation_noise=True).variance.mean(dim=0).cpu().detach()  # type: ignore

            preds, variance = self.output_transform.untransform(preds, variance)
            preds = preds.numpy()
            stds = np.sqrt(variance.numpy())  # type: ignore
        return preds, stds

    def _dumps(self) -> str:
        """Dumps the actual model to a string via pickle as this is not directly json serializable."""
        buffer = io.BytesIO()
        torch.save(
            {"model": self.model, "output_transform": self.output_transform},
            buffer,
            pickle_module=dill,
        )
        return base64.b64encode(buffer.getvalue()).decode()

    def loads(self, data: str):
        """Loads the actual model from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        buffer = io.BytesIO(base64.b64decode(data.encode()))
        path = torch.load(buffer, pickle_module=dill)
        self.model = path["model"]
        self.output_transform = path["output_transform"]
