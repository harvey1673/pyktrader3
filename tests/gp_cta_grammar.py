from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import random

import numpy as np
import pandas as pd

from pycmqlib3.analytics.tstool import calc_funda_signal

try:
    from deap import algorithms, base, creator, tools
except Exception:  # pragma: no cover
    algorithms = None
    base = None
    creator = None
    tools = None


@dataclass
class StrategyGenome:
    """A single strategy individual in the GP grammar search space."""

    asset: str
    feature: str
    signal_func: str
    param_rng: Tuple[int, int, int]
    proc_func: str = ""
    post_func: str = ""
    bullish: bool = True
    chg_func: str = "diff"
    freq: str = "price"
    vol_win: int = 120
    signal_cap: Tuple[float, float] = (-2.0, 2.0)

    fitness: float = np.nan
    train_sharpe: float = np.nan
    valid_sharpe: float = np.nan
    turnover: float = np.nan
    n_obs: int = 0

    def complexity(self) -> int:
        """Return a simple complexity score based on chain length."""
        proc_len = 0 if not self.proc_func else len(self.proc_func.split("|"))
        post_len = 0 if not self.post_func else len(self.post_func.split("|"))
        return 1 + proc_len + post_len

    def to_signal_store_entry(self) -> List[object]:
        """Export as a signal_repo-compatible entry payload."""
        return [
            [self.asset],
            [
                self.feature,
                self.signal_func,
                list(self.param_rng),
                self.proc_func,
                self.chg_func,
                self.bullish,
                self.freq,
                self.post_func,
                self.vol_win,
                list(self.signal_cap),
            ],
        ]


@dataclass
class GrammarSpace:
    """Grammar search space for CTA funda signal evolution."""

    asset_features: Dict[str, List[str]]
    signal_funcs: List[str]
    param_rng_pool: List[Tuple[int, int, int]]
    proc_funcs: List[str] = field(default_factory=lambda: [""])
    post_funcs: List[str] = field(default_factory=lambda: [""])
    bullish_pool: List[bool] = field(default_factory=lambda: [True, False])
    chg_funcs: List[str] = field(default_factory=lambda: ["diff", "pct_change"])
    freq_pool: List[str] = field(default_factory=lambda: ["price", ""])
    vol_win_pool: List[int] = field(default_factory=lambda: [60, 120, 240])
    signal_cap_pool: List[Tuple[float, float]] = field(
        default_factory=lambda: [(-2.0, 2.0), (-2.5, 2.5)]
    )

    def sample_genome(self, rng: random.Random) -> StrategyGenome:
        """Sample one random genome from grammar."""
        assets = list(self.asset_features.keys())
        asset = rng.choice(assets)
        feature = rng.choice(self.asset_features[asset])
        return StrategyGenome(
            asset=asset,
            feature=feature,
            signal_func=rng.choice(self.signal_funcs),
            param_rng=rng.choice(self.param_rng_pool),
            proc_func=rng.choice(self.proc_funcs),
            post_func=rng.choice(self.post_funcs),
            bullish=rng.choice(self.bullish_pool),
            chg_func=rng.choice(self.chg_funcs),
            freq=rng.choice(self.freq_pool),
            vol_win=rng.choice(self.vol_win_pool),
            signal_cap=rng.choice(self.signal_cap_pool),
        )


@dataclass
class GPConfig:
    """Configuration for grammar-based CTA GP evolution."""

    population_size: int = 120
    generations: int = 20
    tournament_size: int = 6
    elite_size: int = 8
    mutation_prob: float = 0.25
    crossover_prob: float = 0.75

    validation_ratio: float = 0.3
    validation_start: Optional[pd.Timestamp] = None
    min_obs: int = 300

    train_weight: float = 0.35
    valid_weight: float = 1.00
    turnover_penalty: float = 0.50
    complexity_penalty: float = 0.03

    risk_scaling: float = 1.0
    cost_rate: float = 2e-4

    random_seed: int = 7


def infer_asset_feature_pool(
    spot_df: pd.DataFrame,
    assets: Sequence[str],
    min_non_na: int = 120,
) -> Dict[str, List[str]]:
    """Infer per-asset feature pool from '<asset>_<feature>' columns."""
    out: Dict[str, List[str]] = {}
    for asset in assets:
        prefix = f"{asset}_"
        feats: List[str] = []
        for col in spot_df.columns:
            if not col.startswith(prefix):
                continue
            if spot_df[col].count() < min_non_na:
                continue
            feats.append(col[len(prefix) :])
        if feats:
            out[asset] = sorted(set(feats))
    return out


class GPCTAEngine:
    """Grammar-based GP engine for CTA feature engineering.

    This version is intentionally simple and focused on the 东证/华泰 style:
    evolve (proc_func, signal_func, post_func, params, feature) on each asset,
    and score by train/validation trading fitness.
    """

    def __init__(
        self,
        spot_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        grammar: GrammarSpace,
        config: Optional[GPConfig] = None,
        bdates: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.spot_df = spot_df.copy()
        self.returns_df = returns_df.copy()
        self.grammar = grammar
        self.config = config or GPConfig()
        self.rng = random.Random(self.config.random_seed)
        self.bdates = bdates

    def _resolve_feature_col(self, genome: StrategyGenome) -> Optional[str]:
        """Resolve feature name to an actual column in spot_df."""
        if genome.feature in self.spot_df.columns:
            return genome.feature
        asset_feature = f"{genome.asset}_{genome.feature}"
        if asset_feature in self.spot_df.columns:
            return asset_feature
        return None

    def _safe_sharpe(self, pnl: pd.Series) -> float:
        if pnl is None or len(pnl) < 30:
            return np.nan
        std = float(pnl.std())
        if std <= 0 or np.isnan(std):
            return np.nan
        return float(np.sqrt(244.0) * pnl.mean() / std)

    def _calc_signal(self, genome: StrategyGenome, feature_col: str) -> Optional[pd.Series]:
        try:
            sig = calc_funda_signal(
                self.spot_df,
                feature_col,
                genome.signal_func,
                list(genome.param_rng),
                proc_func=genome.proc_func,
                chg_func=genome.chg_func,
                bullish=genome.bullish,
                freq=genome.freq,
                signal_cap=list(genome.signal_cap),
                bdates=self.bdates,
                post_func=genome.post_func,
                vol_win=genome.vol_win,
                curr_date=None,
                signal_start=None,
            )
            return sig
        except Exception:
            return None

    def _split_point(self, idx: pd.DatetimeIndex) -> int:
        if self.config.validation_start is not None:
            left = int(np.searchsorted(idx.values, self.config.validation_start.to_datetime64()))
            left = max(1, min(left, len(idx) - 1))
            return left
        left = int(len(idx) * (1.0 - self.config.validation_ratio))
        return max(1, min(left, len(idx) - 1))

    def evaluate_genome(self, genome: StrategyGenome) -> StrategyGenome:
        g = copy.deepcopy(genome)
        feature_col = self._resolve_feature_col(g)
        if feature_col is None or g.asset not in self.returns_df.columns:
            g.fitness = -1e9
            return g

        sig = self._calc_signal(g, feature_col)
        if sig is None or len(sig.dropna()) < self.config.min_obs:
            g.fitness = -1e9
            return g

        ret = self.returns_df[g.asset].dropna()
        sig = sig.reindex(ret.index).ffill()
        data = pd.concat([ret.rename("ret"), sig.rename("sig")], axis=1).dropna()
        if len(data) < self.config.min_obs:
            g.fitness = -1e9
            return g

        vol = data["ret"].rolling(g.vol_win).std().replace(0.0, np.nan)
        holding = (self.config.risk_scaling * data["sig"] / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        trade = holding.diff().abs().fillna(0.0)
        pnl = holding.shift(1).fillna(0.0) * data["ret"] - self.config.cost_rate * trade

        sp = self._split_point(pnl.index)
        train_pnl = pnl.iloc[:sp]
        valid_pnl = pnl.iloc[sp:]

        train_sh = self._safe_sharpe(train_pnl)
        valid_sh = self._safe_sharpe(valid_pnl)
        turnover = float(trade.mean())

        if np.isnan(train_sh):
            train_sh = -5.0
        if np.isnan(valid_sh):
            valid_sh = -5.0

        fit = (
            self.config.train_weight * train_sh
            + self.config.valid_weight * valid_sh
            - self.config.turnover_penalty * turnover
            - self.config.complexity_penalty * g.complexity()
        )

        g.train_sharpe = train_sh
        g.valid_sharpe = valid_sh
        g.turnover = turnover
        g.n_obs = int(len(pnl))
        g.fitness = float(fit)
        return g

    def _tournament_pick(self, population: Sequence[StrategyGenome]) -> StrategyGenome:
        sampled = self.rng.sample(list(population), k=min(self.config.tournament_size, len(population)))
        sampled = sorted(sampled, key=lambda x: x.fitness, reverse=True)
        return copy.deepcopy(sampled[0])

    def _crossover(self, p1: StrategyGenome, p2: StrategyGenome) -> StrategyGenome:
        child = copy.deepcopy(p1)
        keys = [
            "asset",
            "feature",
            "signal_func",
            "param_rng",
            "proc_func",
            "post_func",
            "bullish",
            "chg_func",
            "freq",
            "vol_win",
            "signal_cap",
        ]
        for k in keys:
            if self.rng.random() < 0.5:
                setattr(child, k, copy.deepcopy(getattr(p2, k)))
        return child

    def _mutate(self, g: StrategyGenome) -> StrategyGenome:
        out = copy.deepcopy(g)
        genes = [
            "asset",
            "feature",
            "signal_func",
            "param_rng",
            "proc_func",
            "post_func",
            "bullish",
            "chg_func",
            "freq",
            "vol_win",
            "signal_cap",
        ]
        gene = self.rng.choice(genes)

        if gene == "asset":
            out.asset = self.rng.choice(list(self.grammar.asset_features.keys()))
            out.feature = self.rng.choice(self.grammar.asset_features[out.asset])
        elif gene == "feature":
            feats = self.grammar.asset_features.get(out.asset, [])
            if feats:
                out.feature = self.rng.choice(feats)
        elif gene == "signal_func":
            out.signal_func = self.rng.choice(self.grammar.signal_funcs)
        elif gene == "param_rng":
            out.param_rng = self.rng.choice(self.grammar.param_rng_pool)
        elif gene == "proc_func":
            out.proc_func = self.rng.choice(self.grammar.proc_funcs)
        elif gene == "post_func":
            out.post_func = self.rng.choice(self.grammar.post_funcs)
        elif gene == "bullish":
            out.bullish = self.rng.choice(self.grammar.bullish_pool)
        elif gene == "chg_func":
            out.chg_func = self.rng.choice(self.grammar.chg_funcs)
        elif gene == "freq":
            out.freq = self.rng.choice(self.grammar.freq_pool)
        elif gene == "vol_win":
            out.vol_win = self.rng.choice(self.grammar.vol_win_pool)
        elif gene == "signal_cap":
            out.signal_cap = self.rng.choice(self.grammar.signal_cap_pool)
        return out

    def evolve(self, top_k: int = 20) -> Tuple[pd.DataFrame, List[StrategyGenome]]:
        """Run evolution and return generation history and best genomes."""
        pop = [self.grammar.sample_genome(self.rng) for _ in range(self.config.population_size)]
        history_rows: List[Dict[str, float]] = []

        for gen in range(self.config.generations):
            pop = [self.evaluate_genome(g) for g in pop]
            pop = sorted(pop, key=lambda x: x.fitness, reverse=True)

            fitness_vals = np.array([g.fitness for g in pop], dtype=float)
            history_rows.append(
                {
                    "generation": gen,
                    "best": float(np.nanmax(fitness_vals)),
                    "mean": float(np.nanmean(fitness_vals)),
                    "p90": float(np.nanpercentile(fitness_vals, 90)),
                }
            )

            elites = [copy.deepcopy(g) for g in pop[: self.config.elite_size]]
            next_pop: List[StrategyGenome] = elites

            while len(next_pop) < self.config.population_size:
                p1 = self._tournament_pick(pop)
                p2 = self._tournament_pick(pop)

                if self.rng.random() < self.config.crossover_prob:
                    child = self._crossover(p1, p2)
                else:
                    child = copy.deepcopy(p1)

                if self.rng.random() < self.config.mutation_prob:
                    child = self._mutate(child)
                next_pop.append(child)

            pop = next_pop

        pop = [self.evaluate_genome(g) for g in pop]
        pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
        return pd.DataFrame(history_rows), pop[:top_k]


class DEAPGPCTAEngine(GPCTAEngine):
    """DEAP-backed engine using the same grammar and fitness definitions.

    This keeps the search space and evaluator aligned with the manual engine
    while using deap operators for selection, crossover, and mutation.
    """

    def __init__(
        self,
        spot_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        grammar: GrammarSpace,
        config: Optional[GPConfig] = None,
        bdates: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        if tools is None or creator is None or base is None or algorithms is None:
            raise ImportError(
                "deap is required for DEAPGPCTAEngine. Install with: pip install deap"
            )
        super().__init__(spot_df, returns_df, grammar, config=config, bdates=bdates)
        self._toolbox = self._build_toolbox()

    def _build_toolbox(self):
        if not hasattr(creator, "FitnessMaxCTA"):
            creator.create("FitnessMaxCTA", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualCTA"):
            creator.create("IndividualCTA", list, fitness=creator.FitnessMaxCTA)

        toolbox = base.Toolbox()
        toolbox.register("genome", lambda: self.grammar.sample_genome(self.rng))
        toolbox.register(
            "individual", tools.initRepeat, creator.IndividualCTA, toolbox.genome, n=1
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("clone", copy.deepcopy)
        toolbox.register("evaluate", self._deap_evaluate)
        toolbox.register("mate", self._deap_mate)
        toolbox.register("mutate", self._deap_mutate)
        toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
        return toolbox

    def _deap_evaluate(self, individual):
        individual[0] = self.evaluate_genome(individual[0])
        return (individual[0].fitness,)

    def _deap_mate(self, ind1, ind2):
        c1 = self._crossover(ind1[0], ind2[0])
        c2 = self._crossover(ind2[0], ind1[0])
        ind1[0] = c1
        ind2[0] = c2
        return ind1, ind2

    def _deap_mutate(self, individual):
        individual[0] = self._mutate(individual[0])
        return (individual,)

    def evolve(self, top_k: int = 20) -> Tuple[pd.DataFrame, List[StrategyGenome]]:
        """Run DEAP evolution and return history and top genomes."""
        pop = self._toolbox.population(n=self.config.population_size)
        history_rows: List[Dict[str, float]] = []

        for gen in range(self.config.generations):
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            evaluated = list(map(self._toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, evaluated):
                ind.fitness.values = fit

            pop = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)
            fitness_vals = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
            history_rows.append(
                {
                    "generation": gen,
                    "best": float(np.nanmax(fitness_vals)),
                    "mean": float(np.nanmean(fitness_vals)),
                    "p90": float(np.nanpercentile(fitness_vals, 90)),
                }
            )

            elite_n = min(self.config.elite_size, len(pop))
            elites = [self._toolbox.clone(ind) for ind in pop[:elite_n]]

            offspring_n = max(0, self.config.population_size - elite_n)
            selected = self._toolbox.select(pop, offspring_n)
            offspring = list(map(self._toolbox.clone, selected))

            offspring = algorithms.varAnd(
                offspring,
                self._toolbox,
                cxpb=self.config.crossover_prob,
                mutpb=self.config.mutation_prob,
            )
            pop = elites + offspring

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        evaluated = list(map(self._toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, evaluated):
            ind.fitness.values = fit

        pop = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)
        top = [pop[i][0] for i in range(min(top_k, len(pop)))]
        return pd.DataFrame(history_rows), top


def genomes_to_signal_store(
    genomes: Sequence[StrategyGenome],
    prefix: str = "gp",
) -> Dict[str, List[object]]:
    """Export top genomes to a signal_store-like dictionary."""
    out: Dict[str, List[object]] = {}
    for i, g in enumerate(genomes, start=1):
        out[f"{prefix}_{i:03d}"] = g.to_signal_store_entry()
    return out
