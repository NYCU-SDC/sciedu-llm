set -e

export VERSION="pr-${PR_NUMBER:?PR_NUMBER is required for snapshot cleanup}"

docker compose down -v --rmi all --remove-orphans
