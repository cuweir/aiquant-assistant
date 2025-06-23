import httpx  # For making HTTP requests
import asyncio
import json

# Import rich components
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live  # For potentially dynamic updates if needed later

# !!! IMPORTANT: Update this to your GCP server's IP address !!!
API_BASE_URL = "http://REDACTED-HOST:8000"  # e.g., "http://34.123.45.67:8000"
console = Console()  # Create a Rich console instance


def format_value(value, precision=2):
    """Helper to format numbers, handling None or non-numeric gracefully."""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, (int, str)):  # Allow strings to pass through (like 'N/A')
        return str(value)
    return "N/A"


async def display_analysis_nicely(analysis_data: dict, analysis_id: str = None):
    """Helper function to display a single analysis object using Rich."""

    title_text = "AI Analysis Result"
    if analysis_id:  # analysis_id is the cache key like "BTC/USDT_1h_RSI_OVERSOLD"
        # Try to parse symbol and timeframe from analysis_id if not directly in analysis_data
        # (Though server should ideally include them in the response dict)
        symbol_from_key = analysis_data.get('symbol')
        timeframe_from_key = analysis_data.get('timeframe')

        if not symbol_from_key or not timeframe_from_key:
            try:
                parts = analysis_id.split('_')
                if not symbol_from_key and len(parts) > 0: symbol_from_key = parts[0]
                if not timeframe_from_key and len(parts) > 1: timeframe_from_key = parts[1]
            except:
                pass  # Ignore parsing errors

        title_text = f"AI Analysis (ID: {analysis_id})"
        if symbol_from_key and timeframe_from_key:
            title_text = f"AI Analysis for {symbol_from_key} ({timeframe_from_key})"
        elif symbol_from_key:
            title_text = f"AI Analysis for {symbol_from_key} (ID: {analysis_id})"

    content_table = Table.grid(expand=True, padding=(0, 1))  # Simple grid for key-value
    content_table.add_column(style="dim", min_width=15)  # For labels
    content_table.add_column(ratio=1)  # For values, take remaining space

    content_table.add_row("Timestamp:", Text(str(analysis_data.get('timestamp', 'N/A')), style="cyan"))
    content_table.add_row("Symbol:", Text(str(analysis_data.get('symbol', 'N/A')), style="magenta"))
    content_table.add_row("Timeframe:", Text(str(analysis_data.get('timeframe', 'N/A')), style="magenta"))

    local_signal_text = Text()
    local_signal_text.append(str(analysis_data.get('local_signal', 'N/A')), style="yellow")
    local_signal_text.append(
        f" (RSI: {format_value(analysis_data.get('rsi'))}, Price: {format_value(analysis_data.get('price'))})")
    content_table.add_row("Local Signal:", local_signal_text)

    # Panel for the structured data
    panel_structured_data = Panel(
        content_table,
        title=title_text,
        border_style="green",
        expand=False
    )
    console.print(panel_structured_data)

    # Print AI suggestion separately to handle its Markdown formatting and potential length
    console.print("\n[bold u]AI Suggestion:[/bold u]")
    ai_suggestion_md = analysis_data.get('ai_analysis', 'N/A')
    if ai_suggestion_md and ai_suggestion_md.strip() and ai_suggestion_md != 'N/A':
        # Rich Markdown will render basic markdown to the console.
        # It handles newlines and basic formatting much better.
        # For very long text, the console itself will scroll.
        console.print(Markdown(ai_suggestion_md))
    else:
        console.print(Text("N/A or Empty Suggestion", style="italic dim"))

    console.rule(style="green dim")  # Separator after each full analysis display


async def display_all_analyses():
    async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout a bit
        try:
            with console.status("[bold green]Fetching all cached analyses...[/bold green]", spinner="dots"):
                response = await client.get(f"{API_BASE_URL}/get-all-analyses")
            response.raise_for_status()
            data = response.json()

            console.rule("[bold cyan]--- All Cached AI Analyses ---[/bold cyan]", style="cyan")
            if data.get("analyses") and isinstance(data["analyses"], dict) and data["analyses"]:
                # The server now returns AIAnalysisOutput model dicts, which should be fine.
                # Sort by timestamp (descending, newest first) if possible
                try:
                    sorted_analyses = sorted(
                        data["analyses"].items(),
                        key=lambda item: item[1].get('timestamp', "1970-01-01T00:00:00"),
                        # Default for sorting if missing
                        reverse=True
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Could not sort analyses by timestamp: {e}. Displaying in received order.[/yellow]")
                    sorted_analyses = list(data["analyses"].items())

                for key, analysis in sorted_analyses:
                    await display_analysis_nicely(analysis, analysis_id=key)
                    # console.print("\n") # display_analysis_nicely now adds a rule
            else:
                console.print("No analyses found in cache yet.", style="italic dim")
        except httpx.HTTPStatusError as e:
            console.print(f"[bold red]Error fetching analyses: {e.response.status_code} - {e.response.text}[/bold red]")
        except httpx.RequestError as e:
            console.print(f"[bold red]Request error: {e}[/bold red]")
        except json.JSONDecodeError:
            console.print("[bold red]Error: Could not decode JSON response from server.[/bold red]")


async def manual_trigger(symbol: str, timeframe: str):
    payload = {"symbol": symbol, "timeframe": timeframe}
    # Using a longer timeout for LLM processing on the server
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            with console.status(f"[bold green]Triggering analysis for {symbol} on {timeframe}...[/bold green]",
                                spinner="dots"):
                response = await client.post(f"{API_BASE_URL}/trigger-analysis", json=payload)
            response.raise_for_status()
            analysis_data = response.json()

            console.rule("[bold cyan]--- Manually Triggered AI Analysis ---[/bold cyan]", style="cyan")
            await display_analysis_nicely(analysis_data)  # Re-use the nice display function

        except httpx.HTTPStatusError as e:
            console.print(
                f"[bold red]Error triggering analysis: {e.response.status_code} - {e.response.text}[/bold red]")
        except httpx.RequestError as e:
            console.print(f"[bold red]Request error: {e}[/bold red]")
        except json.JSONDecodeError:
            console.print("[bold red]Error: Could not decode JSON response from server.[/bold red]")


async def main_cli_loop():
    console.print(f"[bold green]AI Quant Assistant CLI - Connecting to: {API_BASE_URL}[/bold green]")
    console.print("Ensure the FastAPI server is running on GCP and accessible.")
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
            symbol_input = console.input("Enter symbol (e.g., BTC/USDT or BTC/USDT:USDT): ")
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
    # Crucial: Check if API_BASE_URL has been updated from the placeholder
    if "YOUR_GCP_VM_EXTERNAL_IP" in API_BASE_URL:
        console.print("[bold red]ERROR: API_BASE_URL in cli.py has not been updated![/bold red]")
        console.print(
            f"Please edit cli.py and replace 'YOUR_GCP_VM_EXTERNAL_IP' in '{API_BASE_URL}' with your server's actual IP address.")
        exit(1)

    try:
        asyncio.run(main_cli_loop())
    except KeyboardInterrupt:
        console.print("\n[yellow]CLI exited by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred in the CLI: {e}[/bold red]")
        # For debugging, you might want to see the full traceback
        # console.print_exception()