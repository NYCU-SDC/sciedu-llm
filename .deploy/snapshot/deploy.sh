set -e

error_handling() {
    cd ~
    if [ -d "$VERSION" ]; then
        cd "$VERSION"
        docker logs "$VERSION"
        docker compose down
        cd ..
        rm -r "$VERSION"
    fi
    exit 1
}

export VERSION="pr-$PR_NUMBER"

enable_error_handling="false"
[ ! -d "$VERSION" ] && enable_error_handling="true"

mkdir -p "$VERSION" || true
envsubst < "./compose.yaml" > "./"$VERSION"/compose.yaml"
cd "$VERSION"

# The service is not up until it *answers*: uvicorn binds its port only after the
# app's lifespan finishes (models check, corpus index build, preset warm), so
# `--wait` is what keeps this script from reporting a successful deploy while
# traefik is still handing out 502s. It waits on the compose healthcheck; the
# timeout is a bound on a startup that hangs, so a stuck index build fails the
# deploy loudly instead of blocking the job forever.
docker compose down
docker compose pull
if [ "$enable_error_handling" == "true" ]; then
    docker compose up -d --wait --wait-timeout 300 || error_handling
else
    docker compose up -d --wait --wait-timeout 300
fi