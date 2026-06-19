import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from pricers import OptionParams, black_scholes_price, binomial_price


def bs_greeks(params: OptionParams) -> dict:
    S, K, r, sigma, T = params.spot, params.strike, params.rate, params.vol, params.T
    q = params.dividend_yield
    is_call = params.option_type == "call"

    if sigma == 0 or T == 0:
        delta = 1.0 if (is_call and S > K) or (not is_call and S < K) else 0.0
        if not is_call:
            delta = -delta if S < K else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        delta = math.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * math.exp(-q * T) * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
            + q * S * math.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -math.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * math.exp(-q * T) * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
            - q * S * math.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)

    theta_per_day = theta / 365.0

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_per_day,
        "rho": rho,
    }


def binomial_greeks(params: OptionParams, n_steps: int = 500) -> dict:
    S = params.spot
    ds = max(S * 0.01, 0.01)

    T = params.T
    if T < 1e-6:
        ds = max(S * 0.005, 0.01)

    dv = 0.01
    if params.vol < 0.01:
        dv = max(params.vol * 0.5 if params.vol > 0 else 0.001, 0.001)

    effective_n = n_steps
    if T < 2 / 365:
        effective_n = max(50, int(T * 365 * 10))

    params_up = OptionParams(
        spot=S + ds,
        strike=params.strike,
        rate=params.rate,
        vol=params.vol,
        days=params.days,
        option_type=params.option_type,
        dividend_yield=params.dividend_yield,
        style=params.style,
    )
    params_down = OptionParams(
        spot=S - ds,
        strike=params.strike,
        rate=params.rate,
        vol=params.vol,
        days=params.days,
        option_type=params.option_type,
        dividend_yield=params.dividend_yield,
        style=params.style,
    )

    p_up = binomial_price(params_up, effective_n)
    p_down = binomial_price(params_down, effective_n)
    p_mid = binomial_price(params, effective_n)

    delta = (p_up - p_down) / (2 * ds)
    gamma = (p_up - 2 * p_mid + p_down) / (ds**2)

    vol_up = max(params.vol + dv, 0.001)
    vol_down = max(params.vol - dv, 0.001)
    params_vol_up = OptionParams(
        spot=S,
        strike=params.strike,
        rate=params.rate,
        vol=vol_up,
        days=params.days,
        option_type=params.option_type,
        dividend_yield=params.dividend_yield,
        style=params.style,
    )
    params_vol_down = OptionParams(
        spot=S,
        strike=params.strike,
        rate=params.rate,
        vol=vol_down,
        days=params.days,
        option_type=params.option_type,
        dividend_yield=params.dividend_yield,
        style=params.style,
    )
    vega = (binomial_price(params_vol_up, effective_n) - binomial_price(params_vol_down, effective_n)) / (2 * dv)

    dt_days = max(1, params.days * 0.01 if params.days >= 1 else 0.1)
    effective_theta_days = max(params.days - dt_days, 0.01)
    params_theta = OptionParams(
        spot=S,
        strike=params.strike,
        rate=params.rate,
        vol=params.vol,
        days=effective_theta_days,
        option_type=params.option_type,
        dividend_yield=params.dividend_yield,
        style=params.style,
    )
    theta = (binomial_price(params_theta, effective_n) - p_mid) / dt_days

    dr = max(0.001, params.rate * 0.1 if params.rate > 0 else 0.001)
    params_r_up = OptionParams(
        spot=S,
        strike=params.strike,
        rate=params.rate + dr,
        vol=params.vol,
        days=params.days,
        option_type=params.option_type,
        dividend_yield=params.dividend_yield,
        style=params.style,
    )
    params_r_down = OptionParams(
        spot=S,
        strike=params.strike,
        rate=max(params.rate - dr, 0),
        vol=params.vol,
        days=params.days,
        option_type=params.option_type,
        dividend_yield=params.dividend_yield,
        style=params.style,
    )
    rho = (binomial_price(params_r_up, effective_n) - binomial_price(params_r_down, effective_n)) / (2 * dr)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def calculate_greeks(params: OptionParams, model: str = "bs") -> dict:
    result = {
        "model": model,
        "warnings": [],
    }

    effective_model = model
    if params.style == "american" and model == "bs":
        effective_model = "binomial"
        result["model"] = "binomial"
        result["warnings"].append(
            f"American options not supported by BS model for Greeks, automatically switched to BINOMIAL"
        )

    if effective_model == "bs":
        greeks = bs_greeks(params)
    elif effective_model == "binomial":
        greeks = binomial_greeks(params)
    else:
        raise ValueError(f"Greeks not supported for model: {model}")

    result.update(greeks)
    return result


def greeks_spot_range(
    params: OptionParams,
    spot_min: float,
    spot_max: float,
    n_points: int = 50,
    model: str = "bs",
) -> pd.DataFrame:
    spots = np.linspace(spot_min, spot_max, n_points)
    rows = []

    for spot in spots:
        p = OptionParams(
            spot=spot,
            strike=params.strike,
            rate=params.rate,
            vol=params.vol,
            days=params.days,
            option_type=params.option_type,
            dividend_yield=params.dividend_yield,
            style=params.style,
        )
        g = calculate_greeks(p, model)
        price = black_scholes_price(p) if model == "bs" else binomial_price(p)
        row = {"spot": spot, "price": price, **g}
        rows.append(row)

    return pd.DataFrame(rows)
