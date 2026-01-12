set -e

docker compose down
docker compose pull
docker compose up -d --wait

echo "finish"