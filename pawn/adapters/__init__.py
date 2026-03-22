"""Adapter methods for fine-tuning frozen PAWN backbones."""

from pawn.adapters.bottleneck import BottleneckAdapter, BottleneckCLM
from pawn.adapters.film import FiLMLayer, FiLMCLM
from pawn.adapters.lora import LoRALinear, LoRACLM
from pawn.adapters.sparse import SparseLinear, SparseCLM
from pawn.adapters.hybrid import HybridCLM
