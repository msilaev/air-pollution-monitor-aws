# Main Terraform configuration for AWS deployment
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "air-pollution-terraform-state"
    key    = "terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  project_name = "air-pollution"
  environment  = var.environment

  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_ecr_repository" "train_service" {
  name = "train-service"
}
resource "aws_ecr_repository" "predict_service" {
  name = "predict-service"
}
resource "aws_ecr_repository" "dashboard_service" {
  name = "dashboard-service"
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = var.s3_bucket_name
  acl    = "private"
}

data "aws_iam_policy_document" "ecs_task_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "ecs_task_role" {
  name               = "ecsTaskRole"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
}
resource "aws_iam_role_policy_attachment" "ecs_s3_access" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_ecs_cluster" "main" {
  name = "mlops-cluster"
}

resource "aws_ecs_task_definition" "train_service" {
  family                   = "train-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_role.arn
  container_definitions    = jsonencode([
    {
      name      = "train-service"
      image     = "${aws_ecr_repository.train_service.repository_url}:latest"
      essential = true
      environment = [
        { name = "AWS_ACCESS_KEY_ID", value = var.aws_access_key_id },
        { name = "AWS_SECRET_ACCESS_KEY", value = var.aws_secret_access_key },
        { name = "AWS_DEFAULT_REGION", value = var.aws_region }
      ]
    }
  ])
}

resource "aws_ecs_service" "dashboard_service" {
  name            = "dashboard-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.train_service.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = true
  }
}

resource "aws_cloudwatch_event_rule" "train_schedule" {
  name                = "train-schedule"
  schedule_expression = "cron(0 8 ? * WED *)"
}
resource "aws_cloudwatch_event_target" "train_target" {
  rule      = aws_cloudwatch_event_rule.train_schedule.name
  arn       = aws_ecs_cluster.main.arn
  role_arn  = aws_iam_role.ecs_task_role.arn
  ecs_target {
    task_definition_arn = aws_ecs_task_definition.train_service.arn
    launch_type         = "FARGATE"
    network_configuration {
      subnets          = var.subnet_ids
      security_groups  = [var.security_group_id]
      assign_public_ip = true
    }
  }
}
