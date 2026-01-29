
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------
model_name = "gpt2"
logger.info(f"Initializing LoRA fine-tuning setup for model: {model_name}")

# --------------------------------------------------
# Load Tokenizer
# --------------------------------------------------
logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
logger.info("Tokenizer loaded successfully")

# --------------------------------------------------
# Load Base Model
# --------------------------------------------------
logger.info("Loading base language model")
model = AutoModelForCausalLM.from_pretrained(model_name)
logger.info("Base model loaded successfully")

# --------------------------------------------------
# LoRA Configuration
# --------------------------------------------------
logger.info("Creating LoRA configuration")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

logger.info(
    f"LoRA config created | r={config.r}, "
    f"alpha={config.lora_alpha}, dropout={config.lora_dropout}"
)

# --------------------------------------------------
# Apply LoRA to Model
# --------------------------------------------------
logger.info("Applying LoRA adapters to the base model")
lora_model = get_peft_model(model, config)

logger.info("LoRA adapters successfully attached to the model")

# --------------------------------------------------
# Optional: Print trainable parameters
# --------------------------------------------------
try:
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())

    logger.info(
        f"Trainable parameters: {trainable_params} / {total_params} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
except Exception as e:
    logger.warning(f"Could not compute trainable parameters: {e}")

