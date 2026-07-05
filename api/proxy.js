// api/proxy.js
// Vercel Serverless Function – همه مدل‌ها را پشتیبانی می‌کند

export default async function handler(req, res) {
  // فقط POST مجاز است
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const { model, prompt, width, height, num_outputs, imageUrl } = req.body;

  if (!prompt) {
    return res.status(400).json({ error: 'پرامپت الزامی است' });
  }

  try {
    let result = null;

    switch (model) {
      case 'stability':
        result = await callStability(prompt, width, height, num_outputs);
        break;
      case 'flux2pro':
        result = await callFlux(prompt, width, height, num_outputs, imageUrl);
        break;
      case 'sdxl':
      case 'sd3':
      case 'nanobanana':
        result = await callHuggingFace(prompt, model);
        break;
      case 'dalle3':
        result = await callDalle3(prompt, width, height, num_outputs);
        break;
      case 'free':
        result = await callFree(prompt);
        break;
      default:
        return res.status(400).json({ error: 'مدل نامعتبر است' });
    }

    if (!result) {
      return res.status(500).json({ error: 'تولید تصویر ناموفق بود' });
    }

    res.json({ output: result });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
}

// ---------- توابع داخلی ----------

async function callStability(prompt, w = 1024, h = 1024, n = 1) {
  const key = process.env.STABILITY_API_KEY;
  if (!key) throw new Error('STABILITY_API_KEY تنظیم نشده است');

  const endpoint = 'https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image';
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${key}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      text_prompts: [{ text: prompt }],
      width: Math.min(w, 1024),
      height: Math.min(h, 1024),
      samples: n
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Stability API: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.artifacts.map(art => 'data:image/png;base64,' + art.base64);
}

async function callFlux(prompt, w = 1024, h = 1024, n = 1, imageUrl = null) {
  const key = process.env.REPLICATE_API_TOKEN;
  if (!key) throw new Error('REPLICATE_API_TOKEN تنظیم نشده است');

  const input = {
    prompt,
    width: Math.min(w, 2048),
    height: Math.min(h, 2048),
    num_outputs: n
  };
  if (imageUrl) input.image = imageUrl;

  // شروع پیش‌بینی
  const startResponse = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: {
      'Authorization': `Token ${key}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      version: 'black-forest-labs/flux-2-pro',
      input
    })
  });

  if (!startResponse.ok) {
    const error = await startResponse.text();
    throw new Error(`Replicate start: ${startResponse.status} - ${error}`);
  }

  const { id } = await startResponse.json();

  // نظرسنجی تا تکمیل شدن (حداکثر ۶۰ ثانیه)
  for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const checkResponse = await fetch(`https://api.replicate.com/v1/predictions/${id}`, {
      headers: { 'Authorization': `Token ${key}` }
    });
    if (!checkResponse.ok) {
      const error = await checkResponse.text();
      throw new Error(`Replicate check: ${checkResponse.status} - ${error}`);
    }
    const statusData = await checkResponse.json();
    if (statusData.status === 'succeeded') {
      return statusData.output;
    }
    if (statusData.status === 'failed') {
      throw new Error('Replicate: تولید ناموفق');
    }
  }

  throw new Error('Replicate: زمان انتظار به پایان رسید');
}

async function callHuggingFace(prompt, modelType) {
  const key = process.env.HF_API_TOKEN;
  if (!key) throw new Error('HF_API_TOKEN تنظیم نشده است');

  let modelId = '';
  switch (modelType) {
    case 'sdxl':
      modelId = 'stabilityai/stable-diffusion-xl-base-1.0';
      break;
    case 'sd3':
      modelId = 'stabilityai/stable-diffusion-3.5-large';
      break;
    case 'nanobanana':
      modelId = 'stabilityai/stable-diffusion-xl-base-1.0'; // جایگزین موقت
      break;
    default:
      throw new Error('مدل Hugging Face نامعتبر است');
  }

  const endpoint = `https://api-inference.huggingface.co/models/${modelId}`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${key}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ inputs: prompt })
  });

  if (response.status === 503) {
    // مدل در حال بارگذاری – صبر کن و دوباره
    await new Promise(r => setTimeout(r, 5000));
    return callHuggingFace(prompt, modelType);
  }

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HF ${modelType}: ${response.status} - ${error}`);
  }

  const buffer = await response.arrayBuffer();
  const base64 = Buffer.from(buffer).toString('base64');
  return `data:image/png;base64,${base64}`;
}

async function callFree(prompt) {
  // بدون کلید – از HF عمومی
  const modelId = 'stabilityai/stable-diffusion-xl-base-1.0';
  const endpoint = `https://api-inference.huggingface.co/models/${modelId}`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: prompt })
  });

  if (response.status === 503) {
    await new Promise(r => setTimeout(r, 5000));
    return callFree(prompt);
  }

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Free model: ${response.status} - ${error}`);
  }

  const buffer = await response.arrayBuffer();
  const base64 = Buffer.from(buffer).toString('base64');
  return `data:image/png;base64,${base64}`;
}

async function callDalle3(prompt, w = 1024, h = 1024, n = 1) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('OPENAI_API_KEY تنظیم نشده است');

  const endpoint = 'https://api.openai.com/v1/images/generations';
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${key}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'dall-e-3',
      prompt,
      n,
      size: `${Math.min(w, 1024)}x${Math.min(h, 1024)}`
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.data.map(item => item.url);
}
