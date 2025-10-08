class BaseDatabaseClient:
	"""Base class for database clients."""

	def __init__(self, config):
		self.config = config

	def connect(self):
		"""Connect to the database."""
		raise NotImplementedError("Subclasses must implement this method.")

	def search(self, query):
		"""Execute a query against the database."""
		raise NotImplementedError("Subclasses must implement this method.")
