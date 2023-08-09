from typing import Annotated, Literal

from pydantic import Extra, Field

from bofire.data_models.kernels.api import AnyKernel, MaternKernel, ScaleKernel
from bofire.data_models.priors.api import (
    BOTORCH_LENGTHCALE_PRIOR,
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
    AnyPrior,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum


class SingleTaskVariationalGPSurrogate(BotorchSurrogate):
    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow

    type: Literal[
        "SingleTaskVariationalGPSurrogate"
    ] = "SingleTaskVariationalGPSurrogate"
    num_outputs: Annotated[int, Field(ge=1)] = 1
    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=MaternKernel(
                ard=True,
                nu=2.5,
                lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR(),
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
