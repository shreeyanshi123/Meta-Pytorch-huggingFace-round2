import asyncio
from server import reset_env

async def main():
    try:
        res = await reset_env()
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
