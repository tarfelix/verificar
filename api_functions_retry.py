# -*- coding: utf-8 -*-
"""
Módulo Cliente HTTP com Resiliência
===================================

Este módulo fornece uma classe, HttpClientRetry, para interagir com a API de
cancelamento de atividades. Ele implementa funcionalidades essenciais para
garantir a robustez das operações:

- Tentativas Automáticas (Retries): Tenta novamente em caso de falhas de rede
  ou erros temporários do servidor (5xx).
- Backoff Exponencial: Aumenta o tempo de espera entre as tentativas para não
  sobrecarregar a API.
- Limite de Taxa (Rate Limiting): Controla o número de chamadas por segundo
  para respeitar os limites da API.
- Modo de Teste (Dry Run): Permite simular as chamadas de API sem realizar
  modificações reais, útil para validação e testes.
"""
import requests
import time
import logging
from typing import Dict, Any, Optional

class HttpClientRetry:
    def __init__(self,
                 base_url: str,
                 entity_id: int,
                 token: str,
                 calls_per_second: float = 3.0,
                 max_attempts: int = 3,
                 timeout: int = 15,
                 dry_run: bool = False):
        """
        Inicializa o cliente HTTP.

        Args:
            base_url (str): A URL base da API.
            entity_id (int): O ID da entidade a ser usado nas chamadas.
            token (str): O token de autorização (Bearer).
            calls_per_second (float): Máximo de chamadas por segundo.
            max_attempts (int): Número máximo de tentativas para cada chamada.
            timeout (int): Timeout em segundos para as requisições.
            dry_run (bool): Se True, não executa as chamadas reais.
        """
        self.base_url = base_url.rstrip("/")
        self.entity_id = entity_id
        self.token = token
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.dry_run = dry_run
        
        # Para o rate limiting
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call_ts = 0

    def _get_headers(self) -> Dict[str, str]:
        """Monta os cabeçalhos padrão para as requisições."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def _rate_limit(self):
        """Garante que o número de chamadas por segundo não seja excedido."""
        if self.min_interval > 0:
            elapsed = time.time() - self.last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_ts = time.time()

    def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Realiza uma requisição HTTP com lógica de retry e backoff.
        """
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_attempts):
            try:
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    headers=self._get_headers(),
                    json=json_data,
                    timeout=self.timeout
                )
                # Lança uma exceção para erros HTTP (4xx, 5xx)
                response.raise_for_status()
                # Se a resposta for bem-sucedida, retorna o JSON
                return response.json()
            
            except requests.exceptions.HTTPError as e:
                # Para erros de cliente (4xx), não adianta tentar novamente.
                if 400 <= e.response.status_code < 500:
                    logging.error(f"Erro de cliente ({e.response.status_code}) na chamada para {url}. Não haverá nova tentativa. Resposta: {e.response.text}")
                    return {"ok": False, "success": False, "error": f"Client Error: {e.response.status_code}", "message": e.response.text}
                # Para erros de servidor (5xx), fazemos retry.
                logging.warning(f"Erro de servidor ({e.response.status_code}) na tentativa {attempt + 1}/{self.max_attempts} para {url}.")
            
            except requests.exceptions.RequestException as e:
                # Para outros erros de rede (timeout, conexão, etc.)
                logging.warning(f"Erro de conexão na tentativa {attempt + 1}/{self.max_attempts} para {url}: {e}")

            # Se não for a última tentativa, espera antes de tentar novamente (backoff)
            if attempt < self.max_attempts - 1:
                wait_time = (2 ** attempt)  # 1s, 2s, 4s...
                time.sleep(wait_time)

        logging.error(f"Todas as {self.max_attempts} tentativas falharam para a requisição {method} {url}.")
        return None

    def activity_canceled(self, activity_id: str, user_name: str) -> Dict[str, Any]:
        """
        Envia uma requisição para cancelar uma atividade por duplicidade.

        Args:
            activity_id (str): O ID da atividade a ser cancelada.
            user_name (str): O nome do usuário que está realizando a ação.

        Returns:
            dict: A resposta da API ou um dicionário de simulação em caso de dry_run.
        """
        if self.dry_run:
            logging.info(f"[DRY-RUN] Simulado o cancelamento da atividade ID: {activity_id} pelo usuário: {user_name}")
            return {"ok": True, "success": True, "message": "Dry run mode"}

        endpoint = f"activity/{self.entity_id}/activitycanceledduplicate"
        body = {
            "entity_id": self.entity_id,
            "id": activity_id,
            "activity_type_id": 152,  # Conforme especificado no guia
            "user_name": user_name
        }
        
        response = self._make_request(method="PUT", endpoint=endpoint, json_data=body)
        
        # Garante que sempre retornamos um dict com estado de sucesso
        if response is None:
            return {"ok": False, "success": False, "error": "Max retries exceeded"}
            
        # A API pode retornar 'ok' ou 'success', então normalizamos
        if "ok" not in response and "success" not in response:
            response["ok"] = False
            response["success"] = False
            
        return response
