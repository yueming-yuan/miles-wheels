import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys

from aiohttp import web, ClientSession


async def main(log_dir, router_bin):
    captured = asyncio.Future()

    async def endpoint(request):
        if request.method == 'POST' and request.path == '/generate':
            payload = await request.json()
            if not captured.done():
                captured.set_result(payload)
            return web.json_response({'text': 'ok', 'meta_info': {'finish_reason': {'type': 'length'}, 'completion_tokens': 1, 'prompt_tokens': 1}})
        return web.json_response({'model_path': 'test-model', 'served_model_name': 'test-model', 'is_generation': True, 'context_length': 8192, 'max_total_num_tokens': 8192, 'tp_size': 1, 'dp_size': 1})

    app = web.Application()
    app.router.add_route('*', '/{path:.*}', endpoint)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 21667).start()
    label = 'binary' if router_bin else 'wheel'
    command = [router_bin] if router_bin else [sys.executable, '-m', 'sglang_router.launch_router']
    router = subprocess.Popen([
        *command,
        '--host', '127.0.0.1', '--port', '21666',
        '--worker-urls', 'http://127.0.0.1:21667', '--policy', 'random',
        '--worker-startup-check-interval', '1',
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tee = subprocess.Popen(['tee', str(log_dir / f'router-probe-{label}.log')], stdin=router.stdout, stdout=subprocess.DEVNULL)
    router.stdout.close()
    payload = {'input_ids': [1], 'return_indexer_topk': True, 'sampling_params': {'max_new_tokens': 1}, 'lora_path': 'probe@1', 'lora_backfill_paths': {'probe@1': '/personal/probe-adapter'}}
    try:
        async with ClientSession() as client:
            async with asyncio.timeout(60):
                while True:
                    try:
                        async with client.get('http://127.0.0.1:21666/health') as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    await asyncio.sleep(0.5)
            async with client.post('http://127.0.0.1:21666/generate', json=payload) as response:
                body = await response.text()
                print('HTTP', response.status, body, flush=True)
                assert response.status == 200
            forwarded = await asyncio.wait_for(captured, 5)
            print(json.dumps({'sent': payload, 'forwarded': forwarded}, indent=2), flush=True)
            present = forwarded.get('lora_backfill_paths') == payload['lora_backfill_paths']
            assert present, forwarded
            assert forwarded.get('return_indexer_topk') is True, forwarded
            print('backfill field preserved:', present, flush=True)
    finally:
        router.terminate()
        await asyncio.to_thread(router.wait)
        tee.wait()
        await runner.cleanup()


args = argparse.ArgumentParser()
args.add_argument('--log-dir', type=Path, required=True)
args.add_argument('--router-bin')
opts = args.parse_args()
opts.log_dir.mkdir(parents=True, exist_ok=True)
asyncio.run(main(opts.log_dir, opts.router_bin))
