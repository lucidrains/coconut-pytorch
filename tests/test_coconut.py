import pytest
import torch
from coconut_pytorch import Coconut

@pytest.mark.parametrize('checkpoint', (True, False))
@pytest.mark.parametrize('num_latent_tokens', (1, 3))
def test_coconut(
    checkpoint,
    num_latent_tokens,
):
    model = Coconut(
        num_reasoning_steps = 3,
        num_latents_per_step = num_latent_tokens,
        checkpoint = checkpoint,
        transformer = dict(
            num_tokens = 256,
            dim = 512,
            depth = 4
        )
    )

    prompt = torch.randint(0, 256, (2, 1024))
    answer = torch.randint(0, 256, (2, 64))

    loss = model(prompt, answer)
    loss.backward()

    generated = model.generate(prompt, max_length = 64)
    assert generated.shape == (2, 64)

# multi stream version

@pytest.mark.parametrize('checkpoint', (True, False))
@pytest.mark.parametrize('num_latent_tokens', (1, 3))
def test_multi_stream_coconut(
    checkpoint,
    num_latent_tokens,
):
    from coconut_pytorch.multi_stream_coconut import Coconut as MultistreamCoconut

    model = MultistreamCoconut(
        num_reasoning_steps = 3,
        num_latents_per_step = num_latent_tokens,
        checkpoint = checkpoint,
        transformer = dict(
            num_tokens = 256,
            dim = 512,
            depth = 4
        )
    )

    prompt = torch.randint(0, 256, (2, 1024))
    answer = torch.randint(0, 256, (2, 64))

    loss = model(prompt, answer)
    loss.backward()

    generated = model.generate(prompt, max_length = 64)
    assert generated.shape == (2, 64)
