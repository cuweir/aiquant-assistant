import httpx
import asyncio
import json
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from app.core.config import settings

load_dotenv()

API_BASE_URL = f"http://{settings.SERVER_PUBLIC_IP}:5173"

API_PREFIX = "/api/v1"
console = Console()


def format_price_dynamically(price: float) -> str:
    if price is None or not isinstance(price, (int, float)):
        return "N/A"
    if price >= 100:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.3f}"
    elif price >= 0.01:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def format_value(value, precision=2):
    if isinstance(value, float): return f"{value:.{precision}f}"
    if isinstance(value, (int, str)): return str(value)
    return "N/A"


async def display_analysis_nicely(analysis_data: dict, analysis_id: str = None):
    symbol_from_data = analysis_data.get('symbol', 'N/A')
    timeframe_from_data = analysis_data.get('timeframe', 'N/A')
    title_text = f"AI Analysis for {symbol_from_data} ({timeframe_from_data})"

    content_table = Table.grid(expand=True, padding=(0, 1))
    content_table.add_column(style="dim", min_width=25)
    content_table.add_column(ratio=1)

    content_table.add_row("Timestamp:", Text(str(analysis_data.get('timestamp', 'N/A')), style="cyan"))
    content_table.add_row("Symbol:", Text(symbol_from_data, style="magenta"))
    content_table.add_row("Timeframe:", Text(timeframe_from_data, style="magenta"))

    price_str = format_price_dynamically(analysis_data.get('price'))
    local_signal_text = Text()
    local_signal_text.append(str(analysis_data.get('local_signal', 'N/A')), style="yellow")
    local_signal_text.append(f" (RSI: {format_value(analysis_data.get('rsi'))}, Price: {price_str})")
    content_table.add_row("Local Signal:", local_signal_text)

    sl_price = analysis_data.get('stop_loss')
    tp_price = analysis_data.get('take_profit')
    if sl_price is not None:
        content_table.add_row("Suggested Stop Loss:", Text(format_price_dynamically(sl_price), style="bold red"))
    if tp_price is not None:
        content_table.add_row("Suggested Take Profit:", Text(format_price_dynamically(tp_price), style="bold green"))

    panel_structured_data = Panel(
        content_table,
        title=title_text,
        border_style="green",
        expand=False
    )
    console.print(panel_structured_data)

    console.print("\n[bold u]AI Suggestion:[/bold u]")
    ai_suggestion_md = analysis_data.get('ai_analysis', 'N/A')
    if ai_suggestion_md and ai_suggestion_md.strip() and "AI analysis not triggered" not in ai_suggestion_md:
        console.print(Markdown(ai_suggestion_md))
    else:
        console.print(Text(ai_suggestion_md, style="italic dim"))
    console.rule(style="green dim")


async def display_all_analyses():
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            with console.status("[bold green]Fetching all cached analyses...[/bold green]", spinner="dots"):
                url = f"{API_BASE_URL}{API_PREFIX}/get-all-analyses"
                console.print(f"[dim]Requesting: {url}[/dim]")
                response = await client.get(url)
            response.raise_for_status()
            analyses_list = response.json()  # <-- This is now a list

            console.rule("[bold cyan]--- All Cached Analyses (from DB) ---[/bold cyan]", style="cyan")

            # --- MODIFICATION START ---
            # Check if the response is a list and if it's not empty
            if isinstance(analyses_list, list) and analyses_list:
                # The data is already a list of analysis dicts, so we can iterate directly.
                # Sorting by timestamp is still a good idea, as list order isn't guaranteed.
                try:
                    sorted_analyses = sorted(
                        analyses_list,
                        key=lambda item: item.get('timestamp', "1970-01-01T00:00:00"),
                        reverse=True
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Could not sort analyses by timestamp: {e}. Displaying in received order.[/yellow]")
                    sorted_analyses = analyses_list

                for analysis in sorted_analyses:
                    # We don't have a unique key like before, so we can pass None for analysis_id
                    await display_analysis_nicely(analysis, analysis_id=None)
            else:
                console.print("No analyses found in the database yet.", style="italic dim")
            # --- MODIFICATION END ---

        except httpx.HTTPStatusError as e:
            console.print(f"[bold red]Error fetching analyses: {e.response.status_code} - {e.response.text}[/bold red]")
        except httpx.RequestError as e:
            console.print(f"[bold red]Request error: {e}[/bold red]")
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response from server.")


async def manual_trigger(symbol: str, timeframe: str):
    payload = {"symbol": symbol, "timeframe": timeframe}
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            with console.status(f"[bold green]Triggering analysis for {symbol} on {timeframe}...[/bold green]",
                                spinner="dots"):
                url = f"{API_BASE_URL}{API_PREFIX}/trigger-analysis"
                console.print(f"[dim]Requesting: {url}[/dim]")
                response = await client.post(url, json=payload)
            response.raise_for_status()
            analysis_data = response.json()
            console.rule("[bold cyan]--- Manually Triggered AI Analysis ---[/bold cyan]", style="cyan")
            await display_analysis_nicely(analysis_data)
        except httpx.HTTPStatusError as e:
            console.print(
                f"[bold red]Error triggering analysis: {e.response.status_code} - {e.response.text}[/bold red]")
        except httpx.RequestError as e:
            console.print(f"[bold red]Request error: {e}[/bold red]")
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response from server.")


async def main_cli_loop():
    console.print(f"[bold green]AI Quant Assistant CLI - Connecting to: {API_BASE_URL}[/bold green]")
    console.print("Ensure the FastAPI server is running and accessible.")
    console.rule(style="green")
    while True:
        console.print("\n[bold]Options:[/bold]")
        console.print("1. View all cached analyses (from server)")
        console.print("2. Manually trigger analysis for a specific symbol")
        console.print("3. Exit")
        choice = console.input("[bold cyan]Enter choice: [/bold cyan]")
        if choice == '1':
            await display_all_analyses()
        elif choice == '2':
            symbol_input = console.input("Enter symbol (e.g., BTC/USDT or XRP/USDT): ")
            timeframe_input = console.input("Enter timeframe (e.g., 1h): ")
            if symbol_input.strip() and timeframe_input.strip():
                await manual_trigger(symbol_input.strip(), timeframe_input.strip())
            else:
                console.print("[yellow]Symbol and timeframe cannot be empty.[/yellow]")
        elif choice == '3':
            console.print("[yellow]Exiting CLI.[/yellow]")
            break
        else:
            console.print("[red]Invalid choice.[/red]")


if __name__ == "__main__":
    if "YOUR_AWS_LIGHTSAIL_IP" in API_BASE_URL:
        console.print("[bold red]ERROR: API_BASE_URL in cli.py has not been updated![/bold red]")
        console.print("Please edit cli.py and replace 'YOUR_AWS_LIGHTSAIL_IP' with your server's actual IP address.")
        exit(1)
    try:
        asyncio.run(main_cli_loop())
    except KeyboardInterrupt:
        console.print("\n[yellow]CLI exited by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred in the CLI: {e}[/bold red]")