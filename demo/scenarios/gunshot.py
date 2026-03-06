import asyncio
import httpx

async def run():
    async with httpx.AsyncClient() as client:
        r = await client.post("http://localhost:8000/demo/start?scenario=gunshot")
        print(f"Started: {r.json()}")


if __name__ == "__main__":
    asyncio.run(run())
