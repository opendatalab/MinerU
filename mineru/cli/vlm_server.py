import click
import sys

from loguru import logger


def vllm_server():
    from mineru.model.vlm.vllm_server import main
    main()


def lmdeploy_server():
    from mineru.model.vlm.lmdeploy_server import main
    main()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    '-e',
    '--engine',
    'inference_engine',
    type=click.Choice(['auto', 'vllm', 'lmdeploy']),
    default='auto',
    help='Select the inference engine used to accelerate VLM inference, default is "auto".',
)
@click.pass_context
def openai_server(ctx, inference_engine):
    sys.argv = [sys.argv[0]] + ctx.args
    if inference_engine == 'auto':
        try:
            import vllm
            inference_engine = 'vllm'
            logger.info("Using vLLM as the inference engine for VLM server.")
        except ImportError:
            logger.info("vLLM not found, attempting to use LMDeploy as the inference engine for VLM server.")
            try:
                import lmdeploy
                inference_engine = 'lmdeploy'
            # Success message moved after successful import
                logger.info("Using LMDeploy as the inference engine for VLM server.")
            except ImportError:
                logger.error("Neither vLLM nor LMDeploy is installed. Please install at least one of them.")
                sys.exit(1)

    if inference_engine == 'vllm':
        try:
            import vllm
        except ImportError:
            logger.error("vLLM is not installed. Please install vLLM or choose LMDeploy as the inference engine.")
            sys.exit(1)
        vllm_server()
    elif inference_engine == 'lmdeploy':
        try:
            import lmdeploy
        except ImportError:
            logger.error("LMDeploy is not installed. Please install LMDeploy or choose vLLM as the inference engine.")
            sys.exit(1)
        lmdeploy_server()

if __name__ == "__main__":
    openai_server()