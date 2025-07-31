#!/bin/bash
# Build, tag, and push all MLOps images to ECR
# Usage: ./build_and_push_all.sh

set -e

REGION="eu-west-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOS=(mlflow train predict dashboard api)
#REPOS=(dashboard)
DOCKERFILES=(Dockerfile.mlflow Dockerfile.train Dockerfile.predict Dockerfile.dashboard Dockerfile.api)
#DOCKERFILES=(Dockerfile.dashboard)

# 1. Create ECR repositories if they don't exist
echo "Creating ECR repositories if needed..."
for repo in "${REPOS[@]}"; do
  aws ecr describe-repositories --repository-names $repo --region $REGION >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name $repo --region $REGION
  echo "ECR repo ensured: $repo"
done

echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# 2. Build, tag, and push each image
for i in ${!REPOS[@]}; do
  repo=${REPOS[$i]}
  dockerfile=${DOCKERFILES[$i]}
  image="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repo:latest"
  echo "Building $repo from $dockerfile..."
  docker build -t $repo:latest -f infrastructure/docker/$dockerfile .
  docker tag $repo:latest $image
  echo "Pushing $image..."
  docker push $image
done

echo "All images built and pushed to ECR."
