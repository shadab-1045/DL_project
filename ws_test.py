import asyncio
import websockets

async def test():
    uri = "ws://localhost:8000/ws/inference?condition=genuine"
    for i in range(3):
        print(f"Connecting {i}...")
        async with websockets.connect(uri) as ws:
            print("Connected. Closing immediately...")
        print("Disconnected.")
        await asyncio.sleep(0.5)
    print("Reconnect tests passed.")

if __name__ == "__main__":
    asyncio.run(test())
