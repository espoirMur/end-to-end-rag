import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import List, Optional

from src.rag.components.ingestion.pdf_parser.docling_parser import DoclingDocumentParser
from src.rag.components.shared.io import IOManager
from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


def validate_directory(path: Path) -> None:
	"""Ensure directory exists and is writable."""
	try:
		path.mkdir(parents=True, exist_ok=True)
		test_file = path / ".permission_test"
		test_file.touch()
		test_file.unlink()
	except (OSError, PermissionError) as e:
		logger.error(f"Directory {path} is not accessible: {str(e)}")
		raise


def get_pdf_files(input_path: Path, limit: Optional[int] = None) -> List[Path]:
	"""Discover PDF files with error handling."""
	try:
		files = list(input_path.glob("**/*.pdf"))
		if not files:
			logger.warning(f"No PDF files found in {input_path}")
		return files[:limit] if limit else files
	except Exception as e:
		logger.error(f"Error discovering PDF files: {str(e)}")
		raise


async def process_files(
	parser: DoclingDocumentParser, file_names: List[Path], max_concurrency: int
) -> tuple[int, int, List[Path]]:
	"""Process files with progress tracking and error handling."""
	logger.info(
		f"Starting to process {len(file_names)} files with concurrency {max_concurrency}"
	)
	start_time = time.monotonic()

	try:
		success, failure, failed = await parser.parse_documents_async(
			file_names, max_concurrent_tasks=max_concurrency
		)
	except Exception as e:
		logger.error(f"Processing failed: {str(e)}", exc_info=True)
		raise

	duration = time.monotonic() - start_time
	logger.info(
		f"Processed {len(file_names)} files in {duration:.2f} seconds "
		f"({len(file_names)/duration:.2f} files/sec)"
	)
	return success, failure, failed


def report_results(
	success: int, failure: int, failed_files: List[Path], total_files: int
) -> None:
	"""Generate comprehensive processing report."""
	success_rate = (success / total_files) * 100 if total_files else 0

	logger.info("\n=== Processing Summary ===")
	logger.info(f"Total files:    {total_files}")
	logger.info(f"Successful:     {success} ({success_rate:.1f}%)")
	logger.info(f"Failed:         {failure}")

	if failed_files:
		logger.info("\nFailed files:")
		# Show first 10 failures
		for idx, file_path in enumerate(failed_files[:10], 1):
			logger.info(f"{idx}. {file_path}")
		if len(failed_files) > 10:
			logger.info(f"... plus {len(failed_files) - 10} more")


async def async_main() -> None:
	"""Main async workflow for PDF processing."""
	parser = argparse.ArgumentParser(description="Parse PDF documents.")
	parser.add_argument(
		"--limit", type=int, default=None, help="Limit the number of files to parse."
	)
	parser.add_argument(
		"--concurrency",
		type=int,
		default=None,
		help="Max concurrent tasks. Defaults to CPU count for CPU-bound or 32 for I/O-bound.",
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path.home() / "Documents",
		help="Input directory containing PDFs (default: ~/Documents)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path.cwd() / "datasets/docling_parsed",
		help="Output directory (default: ./datasets/docling_parsed)",
	)
	args = parser.parse_args()

	# Validate and prepare directories
	validate_directory(args.input)
	validate_directory(args.output)

	# Get files to process
	file_names = get_pdf_files(args.input, args.limit)
	if not file_names:
		return

	# Set intelligent default concurrency
	max_concurrency = args.concurrency or min(32, (os.cpu_count() or 4) * 4)

	# Initialize and run processing
	io_manager = IOManager(input_document_path=args.input, output_path=args.output)
	pdf_parser = DoclingDocumentParser(io_manager=io_manager, document_parser_kwargs={})

	success, failure, failed = await process_files(
		pdf_parser, file_names, max_concurrency
	)

	report_results(success, failure, failed, len(file_names))


if __name__ == "__main__":
	try:
		asyncio.run(async_main())
	except KeyboardInterrupt:
		logger.info("Processing interrupted by user")
	except Exception as e:
		logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
		raise
