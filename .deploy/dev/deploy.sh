set -e

# The service is not up until it *answers*: uvicorn binds its port only after the
# app's lifespan finishes (models check, corpus index build, preset warm), so
# `--wait` is what keeps this script from reporting a successful deploy while
# traefik is still handing out 502s. It waits on the compose healthcheck; the
# timeout is a bound on a startup that hangs, so a stuck index build fails the
# deploy loudly instead of blocking the job forever.
docker compose down
docker compose pull
docker compose up -d --wait --wait-timeout 300

echo "finish"