import pickle
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class BasketOptionNet(nn.Module):
    """Feedforward network for basket option time-value estimation."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [256, 128, 64], dropout: float = 0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.extend([
            nn.Linear(prev_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        ])

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        return self.network(x)


# ============================================================================
# GARCH MODEL CONTAINER
# ============================================================================

class HestonNandiGARCH_Q:
    """Parameter container for a single-asset Heston-Nandi GARCH(1,1) model."""

    def __init__(self):
        self.omega: float | None = None
        self.alpha: float | None = None
        self.beta: float | None = None
        self.gamma: float | None = None
        self.lambda_: float | None = None
        self.h_unconditional: float | None = None

    def get_params(self) -> dict:
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "lambda": self.lambda_,
            "h_unconditional": self.h_unconditional,
        }


# ============================================================================
# VALUATION SYSTEM
# ============================================================================

class BasketOptionValuationSystem:
    """
    Bundles trained call/put neural networks with their scalers and GARCH
    parameters so the hedging environment can price derivatives at runtime.
    """

    def __init__(self, n_assets: int, raw_weights: list | np.ndarray, weight_type: str = "dollar"):
        self.n_assets = n_assets
        self.raw_weights = np.array(raw_weights)
        self.weight_type = weight_type
        self.weights: np.ndarray | None = None
        self.S0_initial: np.ndarray | None = None
        self.price_normalization_factors: np.ndarray | None = None
        self.garch_models: list[HestonNandiGARCH_Q] = [HestonNandiGARCH_Q() for _ in range(n_assets)]
        self.call_model: BasketOptionNet | None = None
        self.put_model: BasketOptionNet | None = None
        self.scaler_call: dict | None = None
        self.scaler_put: dict | None = None
        self.training_data: dict | None = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def load(filepath: str = "basket_system.pkl") -> "BasketOptionValuationSystem":
        """Deserialise a previously saved system from disk."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        system = BasketOptionValuationSystem(
            n_assets=state["n_assets"],
            raw_weights=np.ones(state["n_assets"]) / state["n_assets"],
            weight_type=state["weight_type"],
        )

        system.weights = state["weights"]
        system.S0_initial = state["S0_initial"]
        system.price_normalization_factors = state.get("price_normalization_factors")
        system.scaler_call = state["scaler_call"]
        system.scaler_put = state["scaler_put"]
        system.training_data = state.get("training_data")

        for i, params in enumerate(state["garch_params"]):
            m = system.garch_models[i]
            m.omega = params["omega"]
            m.alpha = params["alpha"]
            m.beta = params["beta"]
            m.gamma = params["gamma"]
            m.lambda_ = params["lambda"]
            m.h_unconditional = params["h_unconditional"]

        input_dim = state["training_data"]["input_dim"]

        system.call_model = BasketOptionNet(input_dim).float()
        system.call_model.load_state_dict(state["call_model_state"])
        system.call_model.eval()

        system.put_model = BasketOptionNet(input_dim).float()
        system.put_model.load_state_dict(state["put_model_state"])
        system.put_model.eval()

        print(f"✓ System loaded from {filepath}")
        print(f"  - {state['n_assets']} assets, input dim: {input_dim}, weight type: {state['weight_type']}")

        return system


# ============================================================================
# UTILITIES
# ============================================================================

def scaler_to_torch(scaler) -> dict[str, torch.Tensor]:
    """Convert a fitted sklearn StandardScaler to plain torch tensors."""
    return {
        "mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "std": torch.tensor(scaler.scale_, dtype=torch.float32),
    }


def correlation_matrix_to_features(R: np.ndarray) -> np.ndarray:
    """
    Flatten the upper triangle of a correlation matrix into a feature vector.
    For N assets this returns N*(N-1)/2 values.
    """
    n = R.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return R[rows, cols]
