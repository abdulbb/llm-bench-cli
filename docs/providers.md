# Provider Setup

`llm-bench` is built on top of [LiteLLM](https://docs.litellm.ai/docs/), allowing it to support a vast number of LLM providers using a unified format.

## API Keys

You must set the appropriate environment variables for the providers you intend to use.

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
```
**Model format:** `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-3.5-turbo`

### Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
**Model format:** `anthropic/claude-3-5-sonnet-20241022`, `anthropic/claude-3-5-haiku-20241022`, `anthropic/claude-3-opus-20240229`

### Google Gemini
```bash
export GEMINI_API_KEY="AIza..."
# or
export GOOGLE_API_KEY="AIza..."
```
**Model format:** `gemini/gemini-1.5-pro`, `gemini/gemini-1.5-flash`, `gemini/gemini-pro`

### Groq
```bash
export GROQ_API_KEY="gsk_..."
```
**Model format:** `groq/llama-3.1-70b-versatile`, `groq/llama-3.1-8b-instant`, `groq/mixtral-8x7b-32768`

Groq offers extremely fast inference speeds, making it ideal for latency-sensitive benchmarks.

### Mistral
```bash
export MISTRAL_API_KEY="..."
```
**Model format:** `mistral/mistral-large-latest`, `mistral/mistral-medium-latest`, `mistral/mistral-small-latest`

### OpenRouter
```bash
export OPENROUTER_API_KEY="sk-or-..."
```
**Model format:** `openrouter/provider/model-name`

OpenRouter provides access to many models through a single API, including free tiers:
- `openrouter/google/gemma-2-9b-it:free`
- `openrouter/meta-llama/llama-3-8b-instruct:free`
- `openrouter/mistralai/mistral-7b-instruct:free`

### Cohere
```bash
export COHERE_API_KEY="..."
```
**Model format:** `cohere/command-r-plus`, `cohere/command-r`, `cohere/command`

### Together AI
```bash
export TOGETHER_API_KEY="..."
```
**Model format:** `together_ai/model-name`

Together AI offers a wide range of open-source models at competitive prices.

### Azure OpenAI
```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-01"
```
**Model format:** `azure/your-deployment-name`

Azure deployments require additional configuration. See [LiteLLM Azure docs](https://docs.litellm.ai/docs/providers/azure) for details.

## Using .env Files

Instead of exporting environment variables, you can use `.env` files:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

LLM-Bench automatically loads:
1. `.env.local` (higher priority)
2. `.env`

Or specify a custom file:
```bash
llm-bench --env-file production.env run --config bench.config.yaml
```

## Common Model Names

| Provider | Model ID | Notes |
| :--- | :--- | :--- |
| **OpenAI** | `openai/gpt-4o` | Latest flagship |
| | `openai/gpt-4o-mini` | Cost-effective |
| | `openai/gpt-4-turbo` | Previous flagship |
| | `openai/gpt-3.5-turbo` | Legacy, very cheap |
| **Anthropic** | `anthropic/claude-3-5-sonnet-20241022` | Best balance |
| | `anthropic/claude-3-5-haiku-20241022` | Fast & cheap |
| | `anthropic/claude-3-opus-20240229` | Most capable |
| **Google** | `gemini/gemini-1.5-pro` | Best for complex tasks |
| | `gemini/gemini-1.5-flash` | Fast and efficient |
| **Groq** | `groq/llama-3.1-70b-versatile` | Fast large model |
| | `groq/llama-3.1-8b-instant` | Ultra-fast small model |
| | `groq/mixtral-8x7b-32768` | Mixture of experts |
| **Mistral** | `mistral/mistral-large-latest` | Most capable |
| | `mistral/mistral-small-latest` | Fast and cheap |
| **OpenRouter** | `openrouter/google/gemma-2-9b-it:free` | Free tier |
| | `openrouter/meta-llama/llama-3-8b-instruct:free` | Free tier |

## Checking API Key Status

Use the `models` command to see which providers are configured:

```bash
llm-bench models
```

Output shows `(configured)` or `(not set)` for each provider.

## Other Providers

For other providers (AWS Bedrock, Vertex AI, HuggingFace, Replicate, etc.), please refer to the [LiteLLM Providers Documentation](https://docs.litellm.ai/docs/providers).

LiteLLM supports 100+ providers. Any model supported by LiteLLM can be used with `llm-bench` by setting the appropriate environment variables.

## Troubleshooting

### Authentication Errors

If you see authentication errors:
1. Verify your API key is correct
2. Check the environment variable name matches the provider
3. Ensure the key has the necessary permissions

```bash
# Debug: Check if key is set
echo $OPENAI_API_KEY | head -c 10
```

### Rate Limits

If you hit rate limits:
1. Reduce `--concurrency` (e.g., `--concurrency 2`)
2. Use `--max-cost` to limit total requests
3. Consider using multiple provider accounts

### Model Not Found

If a model isn't found:
1. Check the exact model ID in provider documentation
2. Verify the model is available in your region/account
3. Use `llm-bench models -p provider` to see known models
