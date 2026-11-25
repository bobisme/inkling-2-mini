// Measure decode speed through an OpenAI-compatible endpoint, the way pi sees it (streaming).
// Usage: bun bench.ts <base-url> [runs=3] [max_tokens=768]
//   bun bench.ts https://<pod>-8000.proxy.runpod.net/v1
// Decode tok/s = completion tokens / (last token - first token). TTFT includes the network.
// Prompts are code-heavy on purpose: DSpark and n-gram drafts are accepted most on code.
export {};

const base = (process.argv[2] ?? "http://127.0.0.1:8080/v1").replace(/\/$/, "");
const runs = Number(process.argv[3] ?? 3);
const maxTokens = Number(process.argv[4] ?? 768);
const keyFile = Bun.file(`${process.env.HOME}/.config/inkling/api-key`);
const apiKey = (await keyFile.exists()) ? (await keyFile.text()).trim() : "none";

const prompts = [
  "Write a TypeScript module implementing an LRU cache with get/set/delete, TTL expiry, and an onEvict callback. Include JSDoc and a short usage example.",
  "Write a Python function that parses ISO-8601 date strings without external libraries, with a docstring and five doctests.",
  "Refactor this into idiomatic Rust with error handling via thiserror, then show unit tests:\n\nfn load(p: &str) -> Vec<String> { std::fs::read_to_string(p).unwrap().lines().map(|l| l.to_string()).collect() }",
];

async function once(prompt: string) {
  const t0 = performance.now();
  const res = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: "inkling",
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature: 0.6,
      stream: true,
      stream_options: { include_usage: true },
      chat_template_kwargs: { reasoning_effort: "low" },
    }),
  });
  if (!res.ok || !res.body) throw new Error(`${res.status} ${await res.text()}`);
  let first = 0, last = 0, tokens = 0, chunks = 0, buf = "";
  const decoder = new TextDecoder();
  for await (const part of res.body) {
    buf += decoder.decode(part, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:") || line === "data: [DONE]") continue;
      const msg = JSON.parse(line.slice(5));
      const d = msg.choices?.[0]?.delta;
      if (d && (d.content || d.reasoning_content || d.reasoning)) {
        const now = performance.now();
        if (!first) first = now;
        last = now;
        chunks++;
      }
      if (msg.usage?.completion_tokens) tokens = msg.usage.completion_tokens;
    }
  }
  const n = tokens || chunks;
  return { ttft: (first - t0) / 1000, tps: (n - 1) / ((last - first) / 1000), n };
}

console.log(`endpoint ${base}`);
const rates: number[] = [];
for (let i = 0; i < runs; i++) {
  const r = await once(prompts[i % prompts.length]!);
  rates.push(r.tps);
  console.log(`run ${i + 1}: ${r.n} tokens, TTFT ${r.ttft.toFixed(2)} s, decode ${r.tps.toFixed(1)} tok/s`);
}
rates.sort((a, b) => a - b);
console.log(`median decode: ${rates[Math.floor(rates.length / 2)]!.toFixed(1)} tok/s`);
