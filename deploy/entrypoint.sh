#!/bin/bash
set -e

# Inject LICHESS_TOKEN into config if provided as env var
if [ -n "$LICHESS_TOKEN" ]; then
    sed -i "s/LICHESS_TOKEN_PLACEHOLDER/$LICHESS_TOKEN/" config.yml
fi

# Start a simple HTTP health check server in background for Cloud Run
# Cloud Run requires a listening port, but lichess-bot doesn't serve HTTP
python3 -c "
import http.server
import socketserver
import threading

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')
    def log_message(self, format, *args):
        pass  # Suppress logs

PORT = int('${PORT:-8080}')
with socketserver.TCPServer(('', PORT), HealthHandler) as httpd:
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f'Health check server on port {PORT}')
" &

# Start lichess-bot
exec python3 lichess-bot.py
