
import logging
from transformers import pipeline

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
# Load Toxicity Classification Model
# --------------------------------------------------
logger.info("Loading toxicity detection model: unitary/toxic-bert")

toxicity = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

logger.info("Toxicity detection model loaded successfully")

# --------------------------------------------------
# Safety Check Function
# --------------------------------------------------
def is_safe(text: str) -> bool:
    """
    Checks whether the given text is safe (non-toxic).

    Returns:
        True  -> Safe content
        False -> Toxic content detected
    """
    if not text or not text.strip():
        logger.warning("Empty or blank text received for safety check")
        return True  # Treat empty input as safe

    logger.info("Running safety check on user input")

    try:
        scores = toxicity(text)
    except Exception as e:
        logger.error(f"Safety model inference failed: {e}")
        return False

    for result in scores:
        label = result.get("label", "").lower()
        score = result.get("score", 0.0)

        logger.info(f"Safety result → label: {label}, score: {score:.4f}")

        if label == "toxic" and score > 0.5:
            logger.warning("🚫 Toxic content detected")
            return False

    logger.info("✅ Content passed safety check")
    return True

# --------------------------------------------------
# Standalone Testing
# --------------------------------------------------
if __name__ == "__main__":
    logger.info("Running safety module standalone test")

    test_1 = "I hate you!"
    test_2 = "AI is amazing"

    logger.info(f"Test input: {test_1}")
    print(is_safe(test_1))  # Expected: False

    logger.info(f"Test input: {test_2}")
    print(is_safe(test_2))  # Expected: True
