import requests

class HttpClient:
    def __init__(self, base_url, entity_id, token, headers=None, timeout=10):
        self.base_url = base_url.rstrip("/")
        self.entity_id = entity_id
        self.headers = headers or f'{{"Content-Type": "application/json", "Authorization": "Bearer {token}"}}'
        self.timeout = timeout

    def activity_canceled(self, activity_id, user_name):
        try:
            import json
            body = f'{{"entity_id":{self.entity_id}, "id":{activity_id}, "activity_type_id":152, "user_name":"{user_name}"}}'
            headers_dict = json.loads(self.headers) if self.headers else None
            body_dict = json.loads(body) if body else None
        except json.JSONDecodeError:
            raise Exception("Invalid JSON in headers or body.")
        else:
            # return f"{self.base_url}/activity/{self.entity_id}/foo"
            response = self._make_request(
                method="PUT",
                url=f"{self.base_url}/activity/{self.entity_id}/activitycanceledduplicate",
                params=None,
                data=body_dict,
                headers=headers_dict
            )

            return response

    def _make_request(self, method, url, params=None, data=None, headers=None):
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return response.text
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return None