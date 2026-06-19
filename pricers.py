import math
import numpy as np
from scipy.stats import norm
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Tuple


class OptionParams(BaseModel):
    spot: float = Field(gt=0, description="Spot price of underlying")
    strike: float = Field(gt=0, description="Strike price")
    rate: float = Field(ge=0, description="Annual risk-free rate (decimal)")
    vol: float = Field(ge=0, description="Annual volatility (decimal)")
    days: float = Field(gt=0, description="Days to maturity")
    option_type: str = Field(default="call", description="call or put")
    dividend_yield: float = Field(default=0.0, ge=0, description="Continuous dividend yield")
    style: str = Field(default="european", description="european or american")

    @field_validator("option_type")
    @classmethod
    def validate_option_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        return v

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        v = v.lower()
        if v not in ("european", "american"):
            raise ValueError("style must be 'european' or 'american'")
        return v

    @property
    def T(self) -> float:
        return self.days / 365.0


def _zero_vol_price(params: OptionParams) -> float:
    S, K, r, T = params.spot, params.strike, params.rate, params.T
    q = params.dividend_yield
    is_call = params.option_type == "call"

    F = S * math.exp((r - q) * T)
    if is_call:
        payoff = max(F - K, 0.0)
    else:
        payoff = max(K - F, 0.0)
    return payoff * math.exp(-r * T)


def black_scholes_price(params: OptionParams) -> float:
    S, K, r, sigma, T = params.spot, params.strike, params.rate, params.vol, params.T
    q = params.dividend_yield

    if sigma == 0 or T == 0:
        return _zero_vol_price(params)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if params.option_type == "call":
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

    return price


def binomial_price(params: OptionParams, n_steps: int = 500) -> float:
    S, K, r, sigma, T = params.spot, params.strike, params.rate, params.vol, params.T
    q = params.dividend_yield
    is_european = params.style == "european"
    is_call = params.option_type == "call"

    if sigma == 0 or T == 0:
        return _zero_vol_price(params)

    dt = T / n_steps
    if dt < 1e-6:
        n_steps = max(10, int(T * 365 * 24))
        dt = T / n_steps

    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)

    if p < 0 or p > 1:
        p = max(0.0, min(1.0, p))

    stock_prices = np.zeros(n_steps + 1)
    option_values = np.zeros(n_steps + 1)

    for i in range(n_steps + 1):
        stock_prices[i] = S * (u ** (n_steps - i)) * (d**i)
        if is_call:
            option_values[i] = max(stock_prices[i] - K, 0.0)
        else:
            option_values[i] = max(K - stock_prices[i], 0.0)

    for j in range(n_steps - 1, -1, -1):
        for i in range(j + 1):
            option_values[i] = math.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            if not is_european:
                stock_price = S * (u ** (j - i)) * (d**i)
                if is_call:
                    intrinsic = stock_price - K
                else:
                    intrinsic = K - stock_price
                option_values[i] = max(option_values[i], intrinsic)

    return option_values[0]


def monte_carlo_price(
    params: OptionParams, n_paths: int = 100000, seed: Optional[int] = 42
) -> Tuple[float, Tuple[float, float]]:
    S, K, r, sigma, T = params.spot, params.strike, params.rate, params.vol, params.T
    q = params.dividend_yield
    is_call = params.option_type == "call"

    if sigma == 0 or T == 0:
        price = _zero_vol_price(params)
        return price, (price, price)

    rng = np.random.default_rng(seed)
    half_paths = n_paths // 2
    Z = rng.standard_normal(half_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)

    ST_positive = S * np.exp(drift + diffusion * Z)
    ST_negative = S * np.exp(drift - diffusion * Z)

    if is_call:
        payoff_pos = np.maximum(ST_positive - K, 0.0)
        payoff_neg = np.maximum(ST_negative - K, 0.0)
    else:
        payoff_pos = np.maximum(K - ST_positive, 0.0)
        payoff_neg = np.maximum(K - ST_negative, 0.0)

    payoffs = np.concatenate([payoff_pos, payoff_neg])
    discounted_payoffs = np.exp(-r * T) * payoffs

    price = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs, ddof=1) / math.sqrt(len(discounted_payoffs))
    z_95 = 1.96
    ci_lower = price - z_95 * std_err
    ci_upper = price + z_95 * std_err

    return price, (ci_lower, ci_upper)


def implied_volatility(
    market_price: float,
    params: OptionParams,
    initial_guess: float = 0.2,
    max_iter: int = 20,
    tolerance: float = 1e-6,
) -> Tuple[Optional[float], dict]:
    info = {
        "iterations": 0,
        "converged": False,
        "final_error": None,
        "history": [],
    }

    if market_price <= 0:
        info["final_error"] = "market price must be positive"
        return None, info

    vol = initial_guess

    for i in range(max_iter):
        info["iterations"] = i + 1
        test_params = OptionParams(
            spot=params.spot,
            strike=params.strike,
            rate=params.rate,
            vol=vol,
            days=params.days,
            option_type=params.option_type,
            dividend_yield=params.dividend_yield,
            style=params.style,
        )
        price = black_scholes_price(test_params)
        error = price - market_price

        info["history"].append({"iter": i + 1, "vol": vol, "price": price, "error": error})

        if abs(error) < tolerance:
            info["converged"] = True
            info["final_error"] = abs(error)
            return vol, info

        vega_bs = _bs_vega(test_params)
        if abs(vega_bs) < 1e-10:
            info["final_error"] = "vega too small, cannot proceed"
            return None, info

        vol = vol - error / vega_bs

        if vol < 0.001:
            vol = 0.001

    info["final_error"] = abs(error)
    return None, info


def _bs_vega(params: OptionParams) -> float:
    S, K, r, sigma, T = params.spot, params.strike, params.rate, params.vol, params.T
    q = params.dividend_yield

    if sigma == 0 or T == 0:
        return 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


def price_option(params: OptionParams, model: str = "bs") -> dict:
    result = {
        "model": model,
        "price": None,
        "ci_lower": None,
        "ci_upper": None,
        "warnings": [],
    }

    effective_model = model
    if params.style == "american" and model in ("bs", "monte-carlo"):
        effective_model = "binomial"
        result["model"] = "binomial"
        result["warnings"].append(
            f"American options not supported by {model.upper()} model, automatically switched to BINOMIAL"
        )

    if effective_model == "bs":
        result["price"] = black_scholes_price(params)
    elif effective_model == "binomial":
        result["price"] = binomial_price(params)
    elif effective_model == "monte-carlo":
        price, ci = monte_carlo_price(params)
        result["price"] = price
        result["ci_lower"] = ci[0]
        result["ci_upper"] = ci[1]
    else:
        raise ValueError(f"Unknown model: {model}")

    return result
