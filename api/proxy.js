// api/proxy.js – نسخه‌ی پایدار با رفع خطاهای Flux و HF

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method Not Allowed' });

  const { model, prompt, width, height, num_outputs, imageUrl } = req.body;

  if (!prompt) return res.status(400).json({ error: 'پرامپت الزامی است' });

  try {
    let result = null;
    switch (model) {
      case 'stability': result = await callStability(prompt, width, height, num_outputs); break;
      case 'flux2pro': result = await callFlux(prompt, width, height, num_outputs, imageUrl); break;
      case 'sdxl': case 'sd3': case 'nanobanana': result = await callHuggingFace(prompt, model); break;
      case 'dalle3': result = await callDalle3(prompt, width, height, num_outputs); break;
      case 'free': result = await callFree(prompt); break;
      default: return res.status(400).json({ error: 'مدل نامعتبر' });
    }

    if (!result || (Array.isArray(result) && result.length === 0)) {
      return res.status(500).json({ error: 'تولید تصویر ناموفق – خروجی خالی' });
    }

    // اگر نتیجه یک آرایه است، لینک‌های معتبر رو فیلتر کن
    if (Array.isArray(result)) {
      result = result.filter(url => url && url.startsWith('http'));
      if (result.length === 0) return res.status(500).json({ error: 'همه لینک‌ها خالی یا نامعتبر هستند' });
    }

    res.json({ output: result });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message || 'خطای ناشناخته' });
  }
}

// ---------- توابع ----------

async function callFlux(prompt, w = 1024, h = 1024, n = 1, imageUrl = null) {
  const key = process.env.REPLICATE_API_TOKEN;
  if (!key) throw new Error('REPLICATE_API_TOKEN تنظیم نشده');

  const input = {
    prompt,
    width: Math.min(w, 2048),
    height: Math.min(h, 2048),
    num_outputs: Math.min(n, 4) // محدود به ۴ تا برای جلوگیری از لینک‌های اضافی
  };
  if (imageUrl) input.image = imageUrl;

  const start = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: { 'Authorization': `Token ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ version: 'black-forest-labs/flux-2-pro', input })
  });
  if (!start.ok) throw new Error(`Replicate start: ${start.status}`);

  const { id } = await start.json();
  for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const check = await fetch(`https://api.replicate.com/v1/predictions/${id}`, {
      headers: { 'Authorization': `Token ${key}` }
    });
    if (!check.ok) throw new Error(`Replicate check: ${check.status}`);
    const data = await check.json();
    if (data.status === 'succeeded') {
      // اگر خروجی آرایه است، برگردان، وگرنه یک آرایه بساز
      if (Array.isArray(data.output)) return data.output;
      else return [data.output];
    }
    if (data.status === 'failed') throw new Error('Replicate: تولید ناموفق');
  }
  throw new Error('Replicate: زمان انتظار تمام شد');
}

async function callHuggingFace(prompt, modelType) {
  const key = process.env.HF_API_TOKEN;
  if (!key) throw new Error('HF_API_TOKEN تنظیم نشده');

  let modelId = '';
  if (modelType === 'sdxl') modelId = 'stabilityai/stable-diffusion-xl-base-1.0';
  else if (modelType === 'sd3') modelId = 'stabilityai/stable-diffusion-3.5-large';
  else if (modelType === 'nanobanana') modelId = 'stabilityai/stable-diffusion-xl-base-1.0';
  else throw new Error('مدل HF نامعتبر');

  const endpoint = `https://api-inference.huggingface.co/models/${modelId}`;
  let retries = 3;
  while (retries--) {
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: prompt })
    });
    if (resp.status === 503) {
      await new Promise(r => setTimeout(r, 5000));
      continue;
    }
    if (!resp.ok) throw new Error(`HF ${modelType}: ${resp.status} - ${await resp.text()}`);
    const buffer = await resp.arrayBuffer();
    const base64 = Buffer.from(buffer).toString('base64');
    return [`data:image/png;base64,${base64}`];
  }
  throw new Error('HF: بارگذاری مدل طولانی شد');
}

async function callFree(prompt) {
  return callHuggingFace(prompt, 'sdxl');
}

// بقیه توابع (Stability, DALL-E) همان قبلی است – برای اختصار حذف شد، اما می‌توانید از نسخه قبلی استفاده کنید.
// اگر نیاز شد، کاملش رو دوباره می‌گذارم.
