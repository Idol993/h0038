import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from pathlib import Path

from pricers import OptionParams, price_option, implied_volatility, black_scholes_price
from greeks import calculate_greeks, greeks_spot_range

console = Console()


def _build_params(spot, strike, rate, vol, days, option_type, dividend_yield, style):
    return OptionParams(
        spot=spot,
        strike=strike,
        rate=rate,
        vol=vol,
        days=days,
        option_type=option_type,
        dividend_yield=dividend_yield,
        style=style,
    )


def _parse_position(value, row_idx: int) -> float:
    if pd.isna(value) or value is None:
        raise ValueError(f"Row {row_idx + 1}: 'position' is empty or missing")
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Row {row_idx + 1}: 'position' value '{value}' is not a valid number")


def _load_portfolio_csv(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    console.print(f"[cyan]Loaded {len(df)} positions from {input_csv}[/cyan]")
    for idx, row in df.iterrows():
        if "position" in df.columns:
            try:
                _parse_position(row.get("position"), idx)
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise click.Abort()
    return df


def _print_warnings(warnings):
    if warnings:
        for w in warnings:
            console.print(f"[yellow]⚠️  {w}[/yellow]")
        console.print()


def _print_price_result(params: OptionParams, result: dict, model: str):
    _print_warnings(result.get("warnings", []))

    table = Table(title="Option Pricing Result", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="dim", width=20)
    table.add_column("Value", justify="right")

    effective_model = result.get("model", model)
    table.add_row("Model", effective_model.upper())
    table.add_row("Option Type", params.option_type.upper())
    table.add_row("Style", params.style.upper())
    table.add_row("Spot Price", f"{params.spot:.4f}")
    table.add_row("Strike Price", f"{params.strike:.4f}")
    table.add_row("Risk-Free Rate", f"{params.rate:.4%}")
    table.add_row("Volatility", f"{params.vol:.4%}")
    table.add_row("Days to Maturity", f"{params.days:.0f}")
    table.add_row("Dividend Yield", f"{params.dividend_yield:.4%}")
    table.add_row("", "")
    table.add_row("[bold green]Price", f"[bold green]{result['price']:.6f}")

    if result.get("ci_lower") is not None:
        table.add_row("95% CI Lower", f"{result['ci_lower']:.6f}")
        table.add_row("95% CI Upper", f"{result['ci_upper']:.6f}")

    console.print(table)


def _print_greeks_result(params: OptionParams, greeks: dict, model: str):
    _print_warnings(greeks.get("warnings", []))

    table = Table(title="Option Greeks", show_header=True, header_style="bold magenta")
    table.add_column("Greek", style="dim", width=15)
    table.add_column("Value", justify="right")
    table.add_column("Note", justify="left")

    effective_model = greeks.get("model", model)
    table.add_row("Model", effective_model.upper(), "")
    table.add_row("Option Type", params.option_type.upper(), "")
    table.add_row("Style", params.style.upper(), "")
    table.add_row("", "", "")

    delta_color = "green" if greeks["delta"] >= 0 else "red"
    table.add_row("Delta", f"[{delta_color}]{greeks['delta']:.6f}[/{delta_color}]", "Price sensitivity")

    gamma_color = "cyan"
    table.add_row("Gamma", f"[{gamma_color}]{greeks['gamma']:.6f}[/{gamma_color}]", "Delta convexity")

    table.add_row(
        "Vega",
        Text(f"{greeks['vega']:.6f}", style="yellow"),
        Text("Vol risk (per 1%)", style="yellow"),
    )

    table.add_row(
        "Theta",
        Text(f"{greeks['theta']:.6f}", style="yellow"),
        Text("Time decay (per day)", style="yellow"),
    )

    rho_color = "blue"
    table.add_row("Rho", f"[{rho_color}]{greeks['rho']:.6f}[/{rho_color}]", "Rate sensitivity")

    console.print(table)


def _print_iv_result(params: OptionParams, iv: float, info: dict, market_price: float):
    table = Table(title="Implied Volatility Result", show_header=True, header_style="bold yellow")
    table.add_column("Item", style="dim", width=25)
    table.add_column("Value", justify="right")

    table.add_row("Market Price", f"{market_price:.6f}")
    table.add_row("Option Type", params.option_type.upper())
    table.add_row("Spot / Strike", f"{params.spot:.2f} / {params.strike:.2f}")
    table.add_row("Risk-Free Rate", f"{params.rate:.4%}")
    table.add_row("Days to Maturity", f"{params.days:.0f}")
    table.add_row("Iterations", f"{info['iterations']}")
    table.add_row("Converged", "[green]Yes[/green]" if info["converged"] else "[red]No[/red]")

    if iv is not None:
        table.add_row("", "")
        table.add_row("[bold yellow]Implied Volatility", f"[bold yellow]{iv:.6%}")
    if info.get("final_error") is not None and info["converged"]:
        table.add_row("Final Error", f"{info['final_error']:.2e}")

    console.print(table)

    if info.get("history"):
        hist_table = Table(
            title="Iteration History",
            show_header=True,
            header_style="bold dim",
            show_lines=False,
        )
        hist_table.add_column("Iter", justify="right", style="dim")
        hist_table.add_column("Vol Guess", justify="right")
        hist_table.add_column("Model Price", justify="right")
        hist_table.add_column("Error", justify="right")

        for entry in info["history"]:
            vol_str = f"{entry['vol']:.6%}"
            price_str = f"{entry['price']:.6f}"
            error_str = f"{entry['error']:.6f}"
            err_color = "green" if abs(entry["error"]) < 0.01 else "white"
            hist_table.add_row(
                str(entry["iter"]),
                vol_str,
                price_str,
                f"[{err_color}]{error_str}[/{err_color}]",
            )

        console.print(hist_table)


@click.group()
def cli():
    """European/American option pricing and Greeks calculator."""
    pass


@cli.command()
@click.option("--spot", type=float, required=True, help="Spot price of underlying")
@click.option("--strike", type=float, required=True, help="Strike price")
@click.option("--rate", type=float, required=True, help="Annual risk-free rate (decimal)")
@click.option("--vol", type=float, required=True, help="Annual volatility (decimal)")
@click.option("--days", type=float, required=True, help="Days to maturity")
@click.option("--type", "option_type", type=click.Choice(["call", "put"]), default="call", help="Option type")
@click.option("--style", type=click.Choice(["european", "american"]), default="european", help="Option style")
@click.option("--dividend-yield", type=float, default=0.0, help="Continuous dividend yield (decimal)")
@click.option(
    "--model",
    type=click.Choice(["bs", "binomial", "monte-carlo"]),
    default="bs",
    help="Pricing model",
)
def price(spot, strike, rate, vol, days, option_type, style, dividend_yield, model):
    """Calculate single option price."""
    params = _build_params(spot, strike, rate, vol, days, option_type, dividend_yield, style)
    result = price_option(params, model)
    _print_price_result(params, result, model)


@cli.command()
@click.option("--spot", type=float, required=True, help="Spot price of underlying")
@click.option("--strike", type=float, required=True, help="Strike price")
@click.option("--rate", type=float, required=True, help="Annual risk-free rate (decimal)")
@click.option("--vol", type=float, required=True, help="Annual volatility (decimal)")
@click.option("--days", type=float, required=True, help="Days to maturity")
@click.option("--type", "option_type", type=click.Choice(["call", "put"]), default="call", help="Option type")
@click.option("--style", type=click.Choice(["european", "american"]), default="european", help="Option style")
@click.option("--dividend-yield", type=float, default=0.0, help="Continuous dividend yield (decimal)")
@click.option(
    "--model",
    type=click.Choice(["bs", "binomial"]),
    default="bs",
    help="Greeks calculation model",
)
@click.option("--spot-range", type=(float, float), default=None, help="Spot range (min max) for risk curve")
@click.option("--n-points", type=int, default=50, help="Number of points for spot range")
@click.option("--output", type=str, default=None, help="Output CSV file for spot range results")
def greeks(spot, strike, rate, vol, days, option_type, style, dividend_yield, model, spot_range, n_points, output):
    """Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho)."""
    params = _build_params(spot, strike, rate, vol, days, option_type, dividend_yield, style)

    if spot_range is not None:
        spot_min, spot_max = spot_range
        df = greeks_spot_range(params, spot_min, spot_max, n_points, model)

        if output:
            df.to_csv(output, index=False)
            console.print(f"[green]Risk curve data saved to {output}[/green]")
        else:
            console.print(Panel.fit(
                f"[bold]Risk Curve: {n_points} points from {spot_min:.2f} to {spot_max:.2f}[/bold]\n"
                f"Use --output to save CSV for plotting.",
                title="Spot Range Mode",
            ))

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 100)
        console.print(df.to_string(index=False))
    else:
        g = calculate_greeks(params, model)
        _print_greeks_result(params, g, model)


@cli.command()
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", type=str, default=None, help="Output CSV file path")
@click.option(
    "--model",
    type=click.Choice(["bs", "binomial", "monte-carlo"]),
    default="bs",
    help="Pricing model",
)
def batch(input_csv, output, model):
    """Batch calculate options from CSV file.

    Expected columns: spot, strike, rate, vol, days, type, style (optional), dividend_yield (optional)
    """
    df = pd.read_csv(input_csv)
    console.print(f"[cyan]Loaded {len(df)} options from {input_csv}[/cyan]")

    results = []
    with console.status(f"[bold green]Calculating with {model} model...", spinner="dots"):
        for _, row in df.iterrows():
            option_type = row.get("type", "call")
            style = row.get("style", "european")
            dividend_yield = row.get("dividend_yield", 0.0)

            params = _build_params(
                spot=row["spot"],
                strike=row["strike"],
                rate=row["rate"],
                vol=row["vol"],
                days=row["days"],
                option_type=option_type,
                dividend_yield=dividend_yield,
                style=style,
            )

            r = price_option(params, model)
            g = calculate_greeks(params, "bs" if model == "monte-carlo" else model)

            results.append({
                "spot": params.spot,
                "strike": params.strike,
                "rate": params.rate,
                "vol": params.vol,
                "days": params.days,
                "type": params.option_type,
                "style": params.style,
                "dividend_yield": params.dividend_yield,
                "price": r["price"],
                "delta": g["delta"],
                "gamma": g["gamma"],
                "vega": g["vega"],
                "theta": g["theta"],
                "rho": g["rho"],
            })

    result_df = pd.DataFrame(results)

    if output:
        result_df.to_csv(output, index=False)
        console.print(f"[green]Results saved to {output}[/green]")
    else:
        table = Table(title=f"Batch Results ({model.upper()} model)", show_header=True, header_style="bold cyan")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Type")
        table.add_column("Spot", justify="right")
        table.add_column("Strike", justify="right")
        table.add_column("Price", justify="right", style="bold green")
        table.add_column("Delta", justify="right")
        table.add_column("Gamma", justify="right")
        table.add_column("Vega", justify="right", style="yellow")
        table.add_column("Theta", justify="right", style="yellow")
        table.add_column("Rho", justify="right")

        for i, row in result_df.iterrows():
            table.add_row(
                str(i + 1),
                row["type"].upper(),
                f"{row['spot']:.2f}",
                f"{row['strike']:.2f}",
                f"{row['price']:.4f}",
                f"{row['delta']:.4f}",
                f"{row['gamma']:.4f}",
                f"{row['vega']:.4f}",
                f"{row['theta']:.4f}",
                f"{row['rho']:.4f}",
            )

        console.print(table)


@cli.command("implied-vol")
@click.option("--market-price", type=float, required=True, help="Market price of the option")
@click.option("--spot", type=float, required=True, help="Spot price of underlying")
@click.option("--strike", type=float, required=True, help="Strike price")
@click.option("--rate", type=float, required=True, help="Annual risk-free rate (decimal)")
@click.option("--days", type=float, required=True, help="Days to maturity")
@click.option("--type", "option_type", type=click.Choice(["call", "put"]), default="call", help="Option type")
@click.option("--dividend-yield", type=float, default=0.0, help="Continuous dividend yield (decimal)")
@click.option("--initial-guess", type=float, default=0.2, help="Initial volatility guess")
@click.option("--max-iter", type=int, default=20, help="Maximum iterations")
def implied_vol(market_price, spot, strike, rate, days, option_type, dividend_yield, initial_guess, max_iter):
    """Calculate implied volatility from market price using Newton-Raphson."""
    params = _build_params(spot, strike, rate, 0.2, days, option_type, dividend_yield, "european")

    iv, info = implied_volatility(
        market_price=market_price,
        params=params,
        initial_guess=initial_guess,
        max_iter=max_iter,
    )

    _print_iv_result(params, iv, info, market_price)


@cli.command()
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", type=str, default=None, help="Output CSV file path for detailed results")
@click.option("--summary-output", type=str, default=None, help="Output CSV file path for portfolio summary")
@click.option(
    "--model",
    type=click.Choice(["bs", "binomial", "monte-carlo"]),
    default="bs",
    help="Pricing model",
)
def portfolio(input_csv, output, summary_output, model):
    """Calculate portfolio risk with position quantities and grouping.

    Expected columns: spot, strike, rate, vol, days, type, style (optional),
    dividend_yield (optional), position (optional, default 1), portfolio (optional)
    """
    df = _load_portfolio_csv(input_csv)

    results = []
    all_warnings = []

    with console.status(f"[bold green]Calculating portfolio with {model} model...", spinner="dots"):
        for idx, row in df.iterrows():
            option_type = row.get("type", "call")
            style = row.get("style", "european")
            dividend_yield = row.get("dividend_yield", 0.0)
            position = _parse_position(row.get("position", 1.0), idx)
            group = row.get("portfolio", "default")

            params = _build_params(
                spot=row["spot"],
                strike=row["strike"],
                rate=row["rate"],
                vol=row["vol"],
                days=row["days"],
                option_type=option_type,
                dividend_yield=dividend_yield,
                style=style,
            )

            r = price_option(params, model)
            greeks_model = "bs" if model == "monte-carlo" else model
            g = calculate_greeks(params, greeks_model)

            all_warnings.extend(r.get("warnings", []))
            all_warnings.extend(g.get("warnings", []))

            results.append({
                "portfolio": group,
                "position_id": idx + 1,
                "spot": params.spot,
                "strike": params.strike,
                "rate": params.rate,
                "vol": params.vol,
                "days": params.days,
                "type": params.option_type,
                "style": params.style,
                "dividend_yield": params.dividend_yield,
                "position": position,
                "price": r["price"],
                "market_value": r["price"] * position,
                "delta_unit": g["delta"],
                "delta_total": g["delta"] * position,
                "gamma_unit": g["gamma"],
                "gamma_total": g["gamma"] * position,
                "vega_unit": g["vega"],
                "vega_total": g["vega"] * position,
                "theta_unit": g["theta"],
                "theta_total": g["theta"] * position,
                "rho_unit": g["rho"],
                "rho_total": g["rho"] * position,
            })

    _print_warnings(all_warnings)

    result_df = pd.DataFrame(results)

    if output:
        result_df.to_csv(output, index=False)
        console.print(f"[green]Detailed results saved to {output}[/green]")

    summary = result_df.groupby("portfolio").agg({
        "market_value": "sum",
        "delta_total": "sum",
        "gamma_total": "sum",
        "vega_total": "sum",
        "theta_total": "sum",
        "rho_total": "sum",
        "position": "count",
    }).rename(columns={"position": "num_positions"}).reset_index()

    total_row = pd.DataFrame([{
        "portfolio": "TOTAL",
        "num_positions": summary["num_positions"].sum(),
        "market_value": summary["market_value"].sum(),
        "delta_total": summary["delta_total"].sum(),
        "gamma_total": summary["gamma_total"].sum(),
        "vega_total": summary["vega_total"].sum(),
        "theta_total": summary["theta_total"].sum(),
        "rho_total": summary["rho_total"].sum(),
    }])
    summary = pd.concat([summary, total_row], ignore_index=True)

    if summary_output:
        summary.to_csv(summary_output, index=False)
        console.print(f"[green]Portfolio summary saved to {summary_output}[/green]")

    sum_table = Table(title="Portfolio Risk Summary", show_header=True, header_style="bold magenta")
    sum_table.add_column("Portfolio", style="bold")
    sum_table.add_column("# Pos", justify="right")
    sum_table.add_column("Market Value", justify="right", style="bold green")
    sum_table.add_column("Total Delta", justify="right")
    sum_table.add_column("Total Gamma", justify="right", style="cyan")
    sum_table.add_column("Total Vega", justify="right", style="yellow")
    sum_table.add_column("Total Theta", justify="right", style="yellow")
    sum_table.add_column("Total Rho", justify="right", style="blue")

    for _, row in summary.iterrows():
        is_total = row["portfolio"] == "TOTAL"
        delta_color = "green" if row["delta_total"] >= 0 else "red"
        if is_total:
            portfolio_name = f"[bold]{row['portfolio']}[/bold]"
        else:
            portfolio_name = row["portfolio"]
        sum_table.add_row(
            portfolio_name,
            str(int(row["num_positions"])),
            f"{row['market_value']:.2f}",
            f"[{delta_color}]{row['delta_total']:.4f}[/{delta_color}]",
            f"{row['gamma_total']:.4f}",
            f"{row['vega_total']:.4f}",
            f"{row['theta_total']:.4f}",
            f"{row['rho_total']:.4f}",
        )

    console.print(sum_table)


@cli.command()
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.option("--spot-shocks", type=str, default="-0.1,-0.05,0,0.05,0.1",
              help="Comma-separated spot price shocks (decimal)")
@click.option("--vol-shocks", type=str, default="-0.05,0,0.05,0.1",
              help="Comma-separated volatility shocks (absolute decimal)")
@click.option("--rate-shocks", type=str, default="-0.01,0,0.01",
              help="Comma-separated interest rate shocks (absolute decimal)")
@click.option("--output", type=str, default=None, help="Output CSV file path for scenario results")
@click.option(
    "--model",
    type=click.Choice(["bs", "binomial"]),
    default="bs",
    help="Pricing model",
)
def scenario(input_csv, spot_shocks, vol_shocks, rate_shocks, output, model):
    """Scenario analysis: compute portfolio PnL and Greeks under market shocks.

    Expected columns: spot, strike, rate, vol, days, type, style (optional),
    dividend_yield (optional), position (optional, default 1), portfolio (optional)
    """
    df = _load_portfolio_csv(input_csv)

    spot_shock_list = [float(x) for x in spot_shocks.split(",")]
    vol_shock_list = [float(x) for x in vol_shocks.split(",")]
    rate_shock_list = [float(x) for x in rate_shocks.split(",")]

    base_portfolio_mv = 0.0
    base_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    positions = []

    with console.status("[bold green]Calculating base scenario...", spinner="dots"):
        for idx, row in df.iterrows():
            option_type = row.get("type", "call")
            style = row.get("style", "european")
            dividend_yield = row.get("dividend_yield", 0.0)
            position = _parse_position(row.get("position", 1.0), idx)

            params = _build_params(
                spot=row["spot"],
                strike=row["strike"],
                rate=row["rate"],
                vol=row["vol"],
                days=row["days"],
                option_type=option_type,
                dividend_yield=dividend_yield,
                style=style,
            )

            r = price_option(params, model)
            g = calculate_greeks(params, model)

            base_portfolio_mv += r["price"] * position
            for gk in base_greeks:
                base_greeks[gk] += g[gk] * position

            positions.append({
                "params": params,
                "position": position,
                "base_price": r["price"],
            })

    scenario_results = []
    total_scenarios = len(spot_shock_list) * len(vol_shock_list) * len(rate_shock_list)

    with console.status(
        f"[bold green]Running {total_scenarios} scenarios...",
        spinner="dots",
    ):
        for ds in spot_shock_list:
            for dv in vol_shock_list:
                for dr in rate_shock_list:
                    port_mv = 0.0
                    port_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
                    vol_clamped_count = 0
                    avg_actual_vol_change = 0.0

                    for pos in positions:
                        p = pos["params"]
                        shocked_spot = p.spot * (1 + ds)
                        raw_vol = p.vol + dv
                        shocked_vol = max(raw_vol, 0.0)
                        shocked_rate = max(p.rate + dr, 0.0)

                        if raw_vol < 0:
                            vol_clamped_count += 1

                        sp = OptionParams(
                            spot=shocked_spot,
                            strike=p.strike,
                            rate=shocked_rate,
                            vol=shocked_vol,
                            days=p.days,
                            option_type=p.option_type,
                            dividend_yield=p.dividend_yield,
                            style=p.style,
                        )

                        r = price_option(sp, model)
                        g = calculate_greeks(sp, model)

                        port_mv += r["price"] * pos["position"]
                        for gk in port_greeks:
                            port_greeks[gk] += g[gk] * pos["position"]

                    pnl = port_mv - base_portfolio_mv
                    pnl_pct = (pnl / base_portfolio_mv * 100) if base_portfolio_mv > 0 else 0.0

                    delta_chg = port_greeks["delta"] - base_greeks["delta"]
                    gamma_chg = port_greeks["gamma"] - base_greeks["gamma"]
                    vega_chg = port_greeks["vega"] - base_greeks["vega"]
                    theta_chg = port_greeks["theta"] - base_greeks["theta"]
                    rho_chg = port_greeks["rho"] - base_greeks["rho"]

                    scenario_results.append({
                        "spot_shock": ds,
                        "vol_shock": dv,
                        "rate_shock": dr,
                        "vol_clamped": vol_clamped_count > 0,
                        "vol_clamped_count": vol_clamped_count,
                        "base_mv": base_portfolio_mv,
                        "portfolio_mv": port_mv,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "base_delta": base_greeks["delta"],
                        "delta": port_greeks["delta"],
                        "delta_chg": delta_chg,
                        "base_gamma": base_greeks["gamma"],
                        "gamma": port_greeks["gamma"],
                        "gamma_chg": gamma_chg,
                        "base_vega": base_greeks["vega"],
                        "vega": port_greeks["vega"],
                        "vega_chg": vega_chg,
                        "base_theta": base_greeks["theta"],
                        "theta": port_greeks["theta"],
                        "theta_chg": theta_chg,
                        "base_rho": base_greeks["rho"],
                        "rho": port_greeks["rho"],
                        "rho_chg": rho_chg,
                    })

    scenario_df = pd.DataFrame(scenario_results)

    base_table = Table(title="Base Portfolio (No Shocks)", show_header=True, header_style="bold cyan")
    base_table.add_column("Metric", style="dim", width=20)
    base_table.add_column("Value", justify="right")
    base_table.add_row("Market Value", f"[bold green]{base_portfolio_mv:.4f}[/bold green]")
    base_table.add_row("Delta", f"{base_greeks['delta']:.4f}")
    base_table.add_row("Gamma", f"{base_greeks['gamma']:.4f}")
    base_table.add_row("Vega", f"[yellow]{base_greeks['vega']:.4f}[/yellow]")
    base_table.add_row("Theta", f"[yellow]{base_greeks['theta']:.4f}[/yellow]")
    base_table.add_row("Rho", f"[blue]{base_greeks['rho']:.4f}[/blue]")
    console.print(base_table)
    console.print()

    worst_idx = scenario_df["pnl"].idxmin()
    best_idx = scenario_df["pnl"].idxmax()

    summary_table = Table(title="Scenario Analysis Summary", show_header=True, header_style="bold yellow")
    summary_table.add_column("Scenario", style="bold", width=10)
    summary_table.add_column("Spot Shock", justify="right")
    summary_table.add_column("Vol Shock", justify="right")
    summary_table.add_column("Rate Shock", justify="right")
    summary_table.add_column("P&L", justify="right")
    summary_table.add_column("ΔDelta", justify="right")
    summary_table.add_column("ΔGamma", justify="right")
    summary_table.add_column("ΔVega", justify="right", style="yellow")
    summary_table.add_column("ΔTheta", justify="right", style="yellow")
    summary_table.add_column("ΔRho", justify="right", style="blue")

    worst = scenario_df.loc[worst_idx]
    best = scenario_df.loc[best_idx]

    pnl_worst_color = "red" if worst["pnl"] < 0 else "green"
    pnl_best_color = "red" if best["pnl"] < 0 else "green"

    def _fmt_chg(val):
        color = "green" if val >= 0 else "red"
        return f"[{color}]{val:+.4f}[/{color}]"

    summary_table.add_row(
        "[red]WORST[/red]",
        f"{worst['spot_shock']:.1%}",
        f"{worst['vol_shock']:.2f}" + (" ⚠" if worst["vol_clamped"] else ""),
        f"{worst['rate_shock']:.2f}",
        f"[{pnl_worst_color}]{worst['pnl']:+.2f}[/{pnl_worst_color}]",
        _fmt_chg(worst["delta_chg"]),
        _fmt_chg(worst["gamma_chg"]),
        _fmt_chg(worst["vega_chg"]),
        _fmt_chg(worst["theta_chg"]),
        _fmt_chg(worst["rho_chg"]),
    )
    summary_table.add_row(
        "[green]BEST[/green]",
        f"{best['spot_shock']:.1%}",
        f"{best['vol_shock']:.2f}" + (" ⚠" if best["vol_clamped"] else ""),
        f"{best['rate_shock']:.2f}",
        f"[{pnl_best_color}]{best['pnl']:+.2f}[/{pnl_best_color}]",
        _fmt_chg(best["delta_chg"]),
        _fmt_chg(best["gamma_chg"]),
        _fmt_chg(best["vega_chg"]),
        _fmt_chg(best["theta_chg"]),
        _fmt_chg(best["rho_chg"]),
    )

    console.print(summary_table)

    clamped_scenarios = scenario_df[scenario_df["vol_clamped"]]
    if len(clamped_scenarios) > 0:
        console.print()
        console.print(f"[yellow]⚠️  {len(clamped_scenarios)} scenario(s) had volatility clamped to 0 for some positions[/yellow]")

    if output:
        scenario_df.to_csv(output, index=False)
        console.print()
        console.print(f"[green]Scenario results saved to {output}[/green]")
        console.print(f"[dim]Total {len(scenario_df)} scenarios calculated[/dim]")
        console.print(f"[dim]Columns include base value, shocked value, and change (_chg) for all metrics[/dim]")


if __name__ == "__main__":
    cli()
