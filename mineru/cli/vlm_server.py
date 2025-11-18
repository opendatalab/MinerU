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
            inference_engine = 'lmdeploy'
            logger.info("vLLM not found, falling back to LMDeploy as the inference engine for VLM server.")
    if inference_engine == 'vllm':
        vllm_server()
    elif inference_engine == 'lmdeploy':
        lmdeploy_server()

if __name__ == "__main__":
    openai_server()