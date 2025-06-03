import argparse

import uvicorn

from src.api.main import app

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Run the FastAPI app with optional reload."
	)
	parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
	args = parser.parse_args()

	uvicorn.run(app, host="0.0.0.0", port=8002, reload=args.reload)
