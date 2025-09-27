import http.server
import socketserver
import requests
import json

# --- CONFIGURAZIONE ---
PORT = 8001
SWISSCOM_API_URL = 'https://api.swisscom.com/layer/swiss-ai-weeks/apertus-70b/v1/chat/completions'

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/proxy':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_body = json.loads(post_data)

                api_key = request_body.get('apiKey')
                payload_to_forward = request_body.get('payload')

                if not api_key or not payload_to_forward:
                    self.send_error(400, "Bad Request: 'apiKey' and 'payload' are required.")
                    return

                # --- LA CORREZIONE È QUI ---
                # Usa l'header standard 'Authorization: Bearer' invece di quello personalizzato.
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
                # -------------------------

                response = requests.post(
                    SWISSCOM_API_URL,
                    headers=headers,
                    json=payload_to_forward,
                    timeout=120
                )
                response.raise_for_status()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response.content)

            except requests.exceptions.RequestException as e:
                # Includi il corpo della risposta di errore per un debug migliore
                error_body = e.response.text if e.response else "No response body"
                self.send_error(502, f"Error from Swisscom API: {e}\nResponse: {error_body}")
            except Exception as e:
                self.send_error(500, f"Internal proxy error: {e}")
        else:
            self.send_error(404, "Endpoint not found.")

# --- ESECUZIONE DEL SERVER ---
if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Server (v2 - Correct Auth) avviato su http://localhost:{PORT}")
        print("Apri http://localhost:8001/dashboard.html nel tuo browser.")
        print("Il server è pronto a servire la dashboard e a fare da proxy per le chiamate API.")
        httpd.serve_forever()