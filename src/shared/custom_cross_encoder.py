import logging
from typing import Dict, Optional

import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.util import fullname, get_device_name, import_from_string
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class CustomCrossEncoder(CrossEncoder):
	def __init__(
		self,
		model_name: str,
		num_labels: int = None,
		max_length: int = None,
		device: str = None,
		tokenizer_args: Dict = None,
		automodel_args: Dict = None,
		trust_remote_code: bool = False,
		revision: Optional[str] = None,
		local_files_only: bool = False,
		default_activation_function=None,
		classifier_dropout: float = None,
		config_kwargs: Dict = None,
	) -> None:
		if tokenizer_args is None:
			tokenizer_args = {}
		if automodel_args is None:
			automodel_args = {}
		self.config = AutoConfig.from_pretrained(
			model_name,
			trust_remote_code=trust_remote_code,
			revision=revision,
			local_files_only=local_files_only,
		)

		for kwarg, value in (config_kwargs or {}).items():
			setattr(self.config, kwarg, value)

		classifier_trained = True
		if self.config.architectures is not None:
			classifier_trained = any(
				[
					arch.endswith("ForSequenceClassification")
					for arch in self.config.architectures
				]
			)

		if classifier_dropout is not None:
			self.config.classifier_dropout = classifier_dropout

		if num_labels is None and not classifier_trained:
			num_labels = 1

		if num_labels is not None:
			self.config.num_labels = num_labels
		self.model = AutoModelForSequenceClassification.from_pretrained(
			model_name,
			config=self.config,
			revision=revision,
			trust_remote_code=trust_remote_code,
			local_files_only=local_files_only,
			**automodel_args,
		)
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_name,
			revision=revision,
			local_files_only=local_files_only,
			trust_remote_code=trust_remote_code,
			**tokenizer_args,
		)
		self.max_length = max_length

		if device is None:
			device = get_device_name()
			logger.info("Use pytorch device: {}".format(device))

		self._target_device = torch.device(device)

		if default_activation_function is not None:
			self.default_activation_function = default_activation_function
			try:
				self.config.sbert_ce_default_activation_function = fullname(
					self.default_activation_function
				)
			except Exception as e:
				logger.warning(
					"Was not able to update config about the default_activation_function: {}".format(
						str(e)
					)
				)
		elif (
			hasattr(self.config, "sbert_ce_default_activation_function")
			and self.config.sbert_ce_default_activation_function is not None
		):
			self.default_activation_function = import_from_string(
				self.config.sbert_ce_default_activation_function
			)()
		else:
			self.default_activation_function = (
				nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()
			)
