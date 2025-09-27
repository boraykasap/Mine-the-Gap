1) run this to build the docker container:
```sh
docker build -t anomaly-workbench .
```

2) Run this command to launch the docker container (make sure to provide your API key)
```sh
docker run --rm -p 8001:8001 -e SWISS_AI_API_KEY=XXXXXXXXX --name anomaly-container anomaly-workbench
```

3) Launch your local webbrowser and navigate to: http://localhost:8001