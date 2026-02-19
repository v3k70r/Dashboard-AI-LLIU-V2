# Dashboard AI‑LLIU (Streamlit)

Este proyecto está pensado para subirse a GitHub **sin** incluir `full_conversations.json` completo (por el límite de 100MB).
En su lugar, el archivo está trozado en partes dentro de `data/`.

## Estructura
- `app.py` — Dashboard Streamlit
- `requirements.txt` — dependencias
- `data/cleaned_cognito_users.csv` — usuarios
- `data/full_conversations.json.part001` ... — conversaciones trozadas (<= 100MB)
- `scripts/split_file.py` — utilidad para trozar cualquier archivo grande

## Ejecutar local
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r requirements.txt
streamlit run app.py
```

> Al iniciar, `app.py` reconstruye automáticamente `data/full_conversations.json` desde las partes `.part###` si no existe.

## Volver a trozar (si cambias el JSON)
```bash
python scripts/split_file.py data/full_conversations.json --max-mib 95 --out-prefix data/full_conversations.json.part
```

Después puedes borrar `data/full_conversations.json` y dejar solo las partes.
