import asyncio
from pathlib import Path

from src.rag.components.ingestion.pdf_scraper.parser import AsyncDocumentParser


async def main():
	async_parser = AsyncDocumentParser()
	documents_path = Path.home().joinpath("Documents")
	file_names = list(documents_path.glob("**/*.pdf"))[:2]

	print(f"need to parse {len(file_names)} files")
	parsing_tasks = []
	for file_name in file_names:
		parsing_task = asyncio.create_task(async_parser.parse_document(file_name))
		parsing_tasks.append(parsing_task)
	await asyncio.gather(*parsing_tasks)


if __name__ == "__main__":
	loop = asyncio.get_event_loop()
	loop.run_until_complete(main())
