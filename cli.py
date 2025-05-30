import httpx  # For making HTTP requests to our FastAPI app
import asyncio
import json
import time

API_BASE_URL = "http://127.0.0.1:8000"  # Default FastAPI server address


async def display_all_analyses():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/get-all-analyses")
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            print("\n--- All Cached AI Analyses ---")
            if data.get("analyses"):
                for key, analysis in data["analyses"].items():
                    print(f"\nAnalysis ID: {key}")
                    print(f"  Timestamp: {analysis.get('timestamp')}")
                    print(f"  Symbol: {analysis.get('symbol')}, Timeframe: {analysis.get('timeframe')}")
                    print(f"  Local Signal: {analysis.get('local_signal')} (RSI: {analysis.get('rsi', 'N/A'):.2f}, Price: {analysis.get('price', 'N/A')})")
                    print(f"  AI Suggestion:\n    {analysis.get('ai_analysis', 'N/A').replace(chr(10), chr(10) + '    ')}")  # Indent multiline
            else:
                print("No analyses found in cache yet.")
        except httpx.HTTPStatusError as e:
            print(f"Error fetching analyses: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            print(f"Request error: {e}")
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response from server.")


async def manual_trigger(symbol: str, timeframe: str):
    payload = {"symbol": symbol, "timeframe": timeframe}
    async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout for LLM
        try:
            print(f"\nTriggering analysis for {symbol} on {timeframe}...")
            response = await client.post(f"{API_BASE_URL}/trigger-analysis", json=payload)
            response.raise_for_status()
            analysis = response.json()
            print("\n--- Manually Triggered AI Analysis ---")
            print(f"  Timestamp: {analysis.get('timestamp')}")
            print(f"  Symbol: {analysis.get('symbol')}, Timeframe: {analysis.get('timeframe')}")
            print(
                f"  Local Signal: {analysis.get('local_signal')} (RSI: {analysis.get('rsi', 'N/A'):.2f}, Price: {analysis.get('price', 'N/A')})")
            print(f"  AI Suggestion:\n    {analysis.get('ai_analysis', 'N/A').replace(chr(10), chr(10) + '    ')}")
        except httpx.HTTPStatusError as e:
            print(f"Error triggering analysis: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            print(f"Request error: {e}")
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response from server.")


async def main_cli_loop():
    print("AI Quant Assistant CLI")
    print("The FastAPI server should be running in the background for this CLI to work.")
    print("The server will periodically check markets based on `SYMBOLS_TO_MONITOR`.")
    print("This CLI primarily displays cached analyses from the server.")
    print("----------------------------------------------------------------")

    while True:
        print("\nOptions:")
        print("1. View all cached analyses (updated by background server task)")
        print("2. Manually trigger analysis for a specific symbol (for testing)")
        print("3. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            await display_all_analyses()
        elif choice == '2':
            symbol = input("Enter symbol (e.g., BTC/USDT or for futures BTC/USDT:USDT): ")
            timeframe = input("Enter timeframe (e.g., 1h): ")
            if symbol and timeframe:
                await manual_trigger(symbol, timeframe)
            else:
                print("Symbol and timeframe cannot be empty.")
        elif choice == '3':
            print("Exiting CLI.")
            break
        else:
            print("Invalid choice.")

        # Add a small delay to prevent spamming the server if in a tight loop (not strictly needed here)
        # await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main_cli_loop())
    except KeyboardInterrupt:
        print("\nCLI exited by user.")