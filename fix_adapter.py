import re

with open("agents.py", "r") as f:
    content = f.read()

old = '''        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()'''

new = '''        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": options.get("temperature", self.temperature),
                "max_tokens": options.get("num_predict", self.num_predict),
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()'''

content = content.replace(old, new)

with open("agents.py", "w") as f:
    f.write(content)

print("Done. Adapter updated.")
