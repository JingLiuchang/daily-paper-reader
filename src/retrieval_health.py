"""Keep an unavailable retrieval source distinct from a valid empty result."""
import os


def strict_retrieval():
    return os.environ.get('DPR_STRICT_RETRIEVAL') == '1'


def require_successful_recall(message):
    if strict_retrieval() and any(marker in message for marker in ('失败', '异常', 'partial_failures=')):
        raise RuntimeError('Paper retrieval failed or was incomplete; refusing to publish a successful daily report: ' + message)
