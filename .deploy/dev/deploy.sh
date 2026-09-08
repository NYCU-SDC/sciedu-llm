set -e

current_sha=$(git rev-parse HEAD)
remote_main_ref=$(git ls-remote --exit-code origin refs/heads/main)
remote_main_sha=${remote_main_ref%%[[:space:]]*}

if [ "$current_sha" != "$remote_main_sha" ]; then
    docker compose down
fi

docker compose pull
docker compose up -d --wait --wait-timeout 300

echo "finish"
