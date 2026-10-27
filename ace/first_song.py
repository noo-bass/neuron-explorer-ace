"""Generate one song end to end with ACE-Step 1.5 on this machine: caption -> planner -> DiT -> VAE -> wav.

Run from this repo with ACE's environment:
    uv run --project ../ACE-Step-1.5 python ace/first_song.py
"""
import os
import time

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

ACE = os.environ.get("ACE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "ACE-Step-1.5"))
ACE = os.path.abspath(ACE)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "ace_first_song"))

t0 = time.time()
dit = AceStepHandler()
dit.initialize_service(project_root=ACE, config_path="acestep-v15-turbo", device="mps")
lm = LLMHandler()
print(lm.initialize(checkpoint_dir=os.path.join(ACE, "checkpoints"), lm_model_path="acestep-5Hz-lm-1.7B",
                    backend="pt", device="mps")[0], flush=True)
print(f"loaded in {time.time() - t0:.0f}s", flush=True)

t0 = time.time()
params = GenerationParams(caption="warm lo-fi hip hop with dusty piano and soft drums", lyrics="[Instrumental]",
                          duration=30, seed=1)
result = generate_music(dit, lm, params, GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False), save_dir=OUT)
print(f"generated in {time.time() - t0:.0f}s", flush=True)
if not result.success:
    raise SystemExit(result.error)
for audio in result.audios:
    print(audio["path"], audio["params"].get("cot_bpm"), audio["params"].get("cot_keyscale"), flush=True)
