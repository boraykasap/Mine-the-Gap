import http.server
import socketserver
import requests
import json
import os

# --- CONFIGURATION ---
PORT = 8001
SWISSCOM_API_URL = 'https://api.swisscom.com/layer/swiss-ai-weeks/apertus-70b/v1/chat/completions'
API_KEY = os.environ.get('SWISS_AI_API_KEY')


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='.')

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
            if not API_KEY:
                self.send_error(500, "Server Error: SWISS_AI_API_KEY environment variable is not set.")
                return

            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                browser_payload = json.loads(post_data)

                # --- LA CORREZIONE È QUI ---
                # Il browser invia solo i messaggi. Il proxy costruisce il payload completo.
                messages_from_browser = browser_payload.get('messages')
                if not messages_from_browser:
                    self.send_error(400, "Bad Request: 'messages' key is missing from the request.")
                    return

                payload_to_forward = {
                    "model": "swiss-ai/Apertus-70B",
                    "messages": messages_from_browser
                }
                # -------------------------

                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {API_KEY}'
                }

                print(f"\n[PROXY] Forwarding payload to Swisscom: {json.dumps(payload_to_forward)}")
                response = requests.post(
                    SWISSCOM_API_URL,
                    headers=headers,
                    json=payload_to_forward,
                    timeout=120
                )

                response.raise_for_status()
                print("[PROXY] Successfully received response from Swisscom.")

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response.content)

            except requests.exceptions.RequestException as e:
                error_body = e.response.text if e.response else "No response body"
                print(f"\n!!! [PROXY] ERROR communicating with Swisscom API: {e}\nResponse Body: {error_body}")
                self.send_error(502, f"Error from Swisscom API: {e}\nResponse: {error_body}")
            except Exception as e:
                print(f"\n!!! [PROXY] INTERNAL ERROR !!!\n{e}")
                self.send_error(500, f"Internal proxy error: {e}")
        else:
            self.send_error(405, "Method Not Allowed")


# --- RUN THE SERVER ---
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Server (v4 - Correct Payload) avviato su http://localhost:{PORT}")
        print(f"Serving files from: {os.getcwd()}")
        print("Open http://localhost:8001/index.html in your browser.")
        httpd.serve_forever()