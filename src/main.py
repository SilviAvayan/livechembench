import asyncio
from ..src.services.pipeline import Pipeline


async def main():

    pipeline = Pipeline()

    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())