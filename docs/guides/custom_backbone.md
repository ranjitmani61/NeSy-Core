# Guide: Custom Neural Backbone

NeSy-Core is **framework-agnostic** — the symbolic layer doesn't know
whether you use PyTorch, TensorFlow, JAX, or a simple numpy model. You
plug in your neural model by subclassing `NeSyBackbone`.

## The Abstract Interface

```python
from nesy.neural.base import NeSyBackbone

class NeSyBackbone(ABC):
    @abstractmethod
    def encode(self, input_data: Any) -> List[float]:
        """Map any input to a fixed-size embedding vector."""
        ...

    @abstractmethod
    def confidence(self, embedding: List[float]) -> float:
        """Return [0, 1] confidence that the embedding is reliable."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings produced by encode()."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    def encode_batch(self, inputs: List[Any]) -> List[List[float]]:
        """Override for efficient batch encoding."""
        return [self.encode(x) for x in inputs]
```

## Example: PyTorch Backbone

```python
import torch
import torch.nn as nn
from nesy.neural.base import NeSyBackbone

class MyTorchBackbone(NeSyBackbone):
    def __init__(self, model: nn.Module, dim: int = 768):
        self._model = model.eval()
        self._dim = dim

    def encode(self, input_data) -> list[float]:
        with torch.no_grad():
            tensor = torch.tensor(input_data).unsqueeze(0)
            embedding = self._model(tensor).squeeze(0)
            return embedding.tolist()

    def confidence(self, embedding: list[float]) -> float:
        mag = sum(x ** 2 for x in embedding) ** 0.5
        return min(1.0, mag / (self._dim ** 0.5))

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "my-torch-backbone"

    def encode_batch(self, inputs: list) -> list[list[float]]:
        with torch.no_grad():
            batch = torch.tensor(inputs)
            embeddings = self._model(batch)
            return embeddings.tolist()
```

## Example: HuggingFace Transformer

```python
from transformers import AutoTokenizer, AutoModel
from nesy.neural.base import NeSyBackbone

class HFBackbone(NeSyBackbone):
    def __init__(self, model_name: str = "bert-base-uncased"):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._dim = self._model.config.hidden_size

    def encode(self, input_data: str) -> list[float]:
        tokens = self._tokenizer(input_data, return_tensors="pt",
                                  truncation=True, max_length=128)
        with torch.no_grad():
            output = self._model(**tokens)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0).tolist()

    def confidence(self, embedding: list[float]) -> float:
        mag = sum(x ** 2 for x in embedding) ** 0.5
        return min(1.0, mag / (self._dim ** 0.5))

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "hf-transformer"
```

## Connecting to the Symbol Grounder

Once you have a backbone, register predicate prototypes with the
`SymbolGrounder` so embeddings can be translated to symbolic predicates:

```python
from nesy.neural.grounding import SymbolGrounder, PredicatePrototype
from nesy.core.types import Predicate

grounder = SymbolGrounder(threshold=0.72)

# Build prototypes from example embeddings
proto = grounder.build_prototype_from_examples(
    predicate=Predicate("fever"),
    example_embeddings=[
        backbone.encode("patient has high temperature"),
        backbone.encode("fever 39.5°C"),
    ],
)
grounder.register(proto)

# Ground a new embedding
symbols = grounder.ground(backbone.encode("temperature elevated"))
for gs in symbols:
    print(f"{gs.predicate} (confidence={gs.grounding_confidence:.3f})")
```

## Using the Neural-Symbolic Bridge

The `NeuralSymbolicBridge` provides bidirectional translation:

```python
from nesy.neural.bridge import NeuralSymbolicBridge

bridge = NeuralSymbolicBridge(grounder=grounder, symbolic_loss_alpha=0.5)

# Neural → Symbolic
predicates, confidence = bridge.neural_to_symbolic(embedding)

# Symbolic → Loss (for training)
loss = bridge.symbolic_to_loss(output_embedding, violated_rules)
```

## Exporting Your Backbone

See the [Edge Deployment Guide](edge_deployment.md) for exporting to
ONNX, TFLite, or CoreML.

## Tips

- Normalise embeddings (L2) for consistent cosine similarity grounding.
- Override `encode_batch()` for GPU-accelerated batching.
- Keep `confidence()` independent of symbolic layer — it measures
  neural model certainty only.
- Use `PassthroughBackbone` for testing without a real ML model.
