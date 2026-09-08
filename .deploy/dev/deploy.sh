set -e

current_sha=$(git rev-parse HEAD)
remote_main_sha=$(git ls-remote origin refs/heads/main | awk '{print $1}')

if [ "$current_sha" != "$remote_main_sha" ]; then
    docker compose down
fi

docker compose pull
docker compose up -d --wait --wait-timeout 300

echo "finish"
