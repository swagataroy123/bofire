import numpy as np
import pandas as pd

import bofire.benchmarks.benchmark as benchmark
import bofire.strategies.api as strategies
from bofire.benchmarks.multi import ZDT1
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import PolytopeSampler
from bofire.data_models.strategies.api import (
    QparegoStrategy as QparegoStrategyDataModel,
)
from bofire.data_models.strategies.api import (
    RejectionSampler as RejectionSamplerDataModel,
)
from bofire.utils.multiobjective import compute_hypervolume


def test_benchmark():
    zdt1 = ZDT1(n_inputs=5)
    qparego_factory = QparegoStrategyDataModel

    n_initial_samples = 10
    n_runs = 3
    n_iterations = 2

    def sample(domain):
        nonlocal n_initial_samples
        sampler = strategies.map(RejectionSamplerDataModel(domain=domain))
        sampled = sampler.ask(n_initial_samples)

        return sampled

    def hypervolume(domain: Domain, experiments: pd.DataFrame) -> float:
        return compute_hypervolume(domain, experiments, ref_point={"y1": 10, "y2": 10})

    results = benchmark.run(
        zdt1,
        strategy_factory=qparego_factory,
        n_iterations=n_iterations,
        metric=hypervolume,
        initial_sampler=sample,
        n_runs=n_runs,
        n_procs=1,
    )

    assert len(results) == n_runs
    for experiments, best in results:
        assert experiments is not None
        assert experiments.shape[0] == n_initial_samples + n_iterations
        assert best.shape[0] == n_iterations
        assert isinstance(best, pd.Series)
        assert isinstance(experiments, pd.DataFrame)


def test_benchmark_generate_outliers():
    def sample(domain):
        datamodel = PolytopeSampler(domain=domain)
        sampler = strategies.map(data_model=datamodel)
        sampled = sampler.ask(10)
        return sampled

    outlier_rate = 0.5
    Benchmark = Himmelblau()
    sampled = sample(Benchmark.domain)
    sampled_xy = Benchmark.f(sampled, return_complete=True)
    Benchmark = Himmelblau(
        outlier_rate=outlier_rate,
        outlier_prior=benchmark.UniformOutlierPrior(bounds=(50, 100)),
    )
    sampled_xy1 = Benchmark.f(sampled, return_complete=True)
    assert np.sum(sampled_xy["y"] != sampled_xy1["y"]) != 0
    assert isinstance(Benchmark.outlier_prior, benchmark.UniformOutlierPrior)
