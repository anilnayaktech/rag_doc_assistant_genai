# from evaluate import load

# rouge = load("rouge")
# predictions = ["AI is intelligence in machines"]
# references = ["Artificial intelligence is the simulation of human intelligence by machines"]

# result = rouge.compute(predictions=predictions, references=references)
# print(result)



#===================================================


import logging
from evaluate import load

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
# Load Evaluation Metric
# --------------------------------------------------
logger.info("Loading ROUGE evaluation metric")
rouge = load("rouge")
logger.info("ROUGE metric loaded successfully")

# --------------------------------------------------
# Sample Predictions & References
# --------------------------------------------------
predictions = ["AI is intelligence in machines"]
references = [
    "Artificial intelligence is the simulation of human intelligence by machines"
]

logger.info("Starting ROUGE evaluation")
logger.debug(f"Predictions: {predictions}")
logger.debug(f"References: {references}")

# --------------------------------------------------
# Compute ROUGE Scores
# --------------------------------------------------
result = rouge.compute(
    predictions=predictions,
    references=references
)

logger.info("ROUGE evaluation completed successfully")
logger.info(f"ROUGE Scores: {result}")

# Optional console output (useful for quick runs)
print("ROUGE Evaluation Result:")
print(result)
