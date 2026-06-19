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


def _print_price_result(params: OptionParams, result: dict, model: str):
    table = Table(title="Option Pricing Result", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="dim", width=20)
    table.add_column("Value", justify="right")

    table.add_row("Model", model.upper())
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
    table = Table(title="Option Greeks", show_header=True, header_style="bold magenta")
    table.add_column("Greek", style="dim", width=15)
    table.add_column("Value", justify="right")
    table.add_column("Note", justify="left")

    table.add_row("Model", model.upper(), "")
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


if __name__ == "__main__":
    cli()
