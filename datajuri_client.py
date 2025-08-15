# -*- coding: utf-8 -*-
"""
Cliente mínimo para API DataJuri com cache:
- OAuth password (client_id/secret + username/password)
- Busca FaseProcesso por número (CNJ)
- Retorna processo.pasta para canalizar a comparação
"""
from __future__ import annotations
import base64, time, requests

class DataJuriClient:
    def __init__(self, base_url: str, client_id: str, secret_id: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.secret_id = secret_id
        self.username = username
        self.password = password
        self._token = None
        self._token_ts = 0
        self._cache = {}

    def ensure_token(self):
        if self._token and (time.time() - self._token_ts) < 50*60:
            return self._token
        auth = f"{self.client_id}:{self.secret_id}"
        b64 = base64.b64encode(auth.encode()).decode()
        url = f"{self.base_url}/oauth/token"
        headers = {'Authorization': f'Basic {b64}', 'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'grant_type': 'password', 'username': self.username, 'password': self.password}
        r = requests.post(url, headers=headers, data=data)
        r.raise_for_status()
        tok = r.json().get("access_token")
        if not tok:
            raise RuntimeError("Token ausente na resposta do DataJuri.")
        self._token = tok; self._token_ts = time.time()
        return tok

    def _headers(self):
        return {"Authorization": f"Bearer {self.ensure_token()}"}

    def get_fase_by_numero(self, numero: str):
        """
        GET /v1/entidades/FaseProcesso?campos=processo.pasta,numero&criterio=numero | igual a | <numero>
        """
        params = [
            ("campos", "processo.pasta,numero"),
            ("pageSize", "1"),
            ("criterio", f"numero | igual a | {numero}")
        ]
        url = f"{self.base_url}/v1/entidades/FaseProcesso"
        r = requests.get(url, headers=self._headers(), params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_pasta_by_cnj_cached(self, cnj: str) -> str | None:
        if not cnj: return None
        if cnj in self._cache: return self._cache[cnj]
        try:
            data = self.get_fase_by_numero(cnj)
            pasta = None
            if data and data.get("rows"):
                row = data["rows"][0]
                # pode vir como "processo.pasta" ou aninhado, dependendo da API
                pasta = row.get("processo.pasta") or row.get("processo",{}).get("pasta")
            self._cache[cnj] = pasta
            return pasta
        except Exception:
            self._cache[cnj] = None
            return None
