import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys

from aiohttp import web, ClientSession

# A response field the router must pass through unchanged (SGLang's Mooncake output store).
OUTPUT_STORE_REF = {'handle': {'type': 'probe', 'version': 1}, 'fields': {'routed_experts': {'dtype': 'int32', 'shape': [1, 2, 3]}}}
GENERATE_RESPONSE = {'text': 'ok', 'meta_info': {'finish_reason': {'type': 'length'}, 'completion_tokens': 1, 'prompt_tokens': 1, 'output_store_ref': OUTPUT_STORE_REF}}
CHAT_RESPONSE = {
    'id': 'probe', 'object': 'chat.completion', 'created': 0, 'model': 'test-model',
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'length', 'meta_info': {'output_store_ref': OUTPUT_STORE_REF}}],
    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
}


async def main(log_dir, router_bin):
    captured = {'/generate': asyncio.Queue(), '/v1/chat/completions': asyncio.Queue()}

    async def endpoint(request):
        if request.method == 'POST' and request.path in captured:
            await captured[request.path].put(await request.json())
            return web.json_response(GENERATE_RESPONSE if request.path == '/generate' else CHAT_RESPONSE)
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
    generate = {'input_ids': [1], 'return_indexer_topk': True, 'sampling_params': {'max_new_tokens': 1}, 'lora_path': 'probe@1', 'lora_backfill_paths': {'probe@1': '/personal/probe-adapter'}}
    chat = {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1, 'return_meta_info': True, 'return_routed_experts': True}

    async def forward(client, path, payload):
        async with client.post(f'http://127.0.0.1:21666{path}', json=payload) as response:
            body = await response.text()
            print(path, 'HTTP', response.status, body, flush=True)
            assert response.status == 200, body
        forwarded = await asyncio.wait_for(captured[path].get(), 5)
        print(json.dumps({'path': path, 'sent': payload, 'forwarded': forwarded}, indent=2), flush=True)
        return json.loads(body), forwarded

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

            _, forwarded = await forward(client, '/generate', generate)
            present = forwarded.get('lora_backfill_paths') == generate['lora_backfill_paths']
            assert present, forwarded
            assert forwarded.get('return_indexer_topk') is True, forwarded
            print('backfill field preserved:', present, flush=True)
            assert 'return_outputs_via_store' not in forwarded, forwarded

            body, forwarded = await forward(client, '/generate', {**generate, 'return_outputs_via_store': True})
            assert forwarded.get('return_outputs_via_store') is True, forwarded
            assert body['meta_info']['output_store_ref'] == OUTPUT_STORE_REF, body

            _, forwarded = await forward(client, '/v1/chat/completions', chat)
            assert 'return_outputs_via_store' not in forwarded, forwarded

            body, forwarded = await forward(client, '/v1/chat/completions', {**chat, 'return_outputs_via_store': True})
            assert forwarded.get('return_outputs_via_store') is True, forwarded
            assert body['choices'][0]['meta_info']['output_store_ref'] == OUTPUT_STORE_REF, body
            print('return_outputs_via_store forwarded only when set; output_store_ref returned on generate and chat', flush=True)
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
