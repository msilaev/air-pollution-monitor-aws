# VPC endpoint for CloudWatch Logs
resource "aws_vpc_endpoint" "cloudwatch_logs" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.logs"
  vpc_endpoint_type = "Interface"
  subnet_ids        = [
    aws_subnet.public_a.id,
    aws_subnet.public_b.id
  ]
  security_group_ids = [
    aws_security_group.cloudwatch_logs_endpoint_sg.id
  ]
  private_dns_enabled = true
}



# Security group for Dashboard ECS service
resource "aws_security_group" "dashboard" {
  name        = "mlops-dashboard-sg"
  description = "Security group for Dashboard ECS service"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port                = 8501
    to_port                  = 8501
    protocol                 = "tcp"
    security_groups          = [aws_security_group.dashboard_alb.id]
    description              = "Allow ALB to access dashboard on port 8501"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
# Restore API ECS Service
resource "aws_ecs_service" "api" {
  name            = "mlops-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
  depends_on = [aws_lb_listener.dashboard]
}





# Allow ALB to reach API ECS task on port 8000
resource "aws_security_group_rule" "allow_alb_to_api" {
  type                     = "ingress"
  from_port                = 8000
  to_port                  = 8000
  protocol                 = "tcp"
  security_group_id        = aws_security_group.api.id
  source_security_group_id = aws_security_group.dashboard_alb.id
  description              = "Allow ALB to reach API ECS task on port 8000"
}
resource "aws_ecs_cluster" "main" {
  name = "mlops-main-cluster"
}



# --- Dedicated Security Groups for Each ECS Resource ---

# Security group for MLflow server
resource "aws_security_group" "mlflow_sg" {
  name        = "mlops-mlflow-sg"
  description = "Security group for MLflow server"
  vpc_id      = aws_vpc.main.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow outbound HTTPS to CloudWatch Logs endpoint
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.private_subnet_a_cidr, var.private_subnet_b_cidr]
  }
}

# Allow train, predict, and API to access MLflow on port 5000
resource "aws_security_group_rule" "mlflow_ingress_train" {
  type                     = "ingress"
  from_port                = 5000
  to_port                  = 5000
  protocol                 = "tcp"
  security_group_id        = aws_security_group.mlflow_sg.id
  source_security_group_id = aws_security_group.train_sg.id
  description              = "Allow train to access MLflow on port 5000"
}
resource "aws_security_group_rule" "mlflow_ingress_predict" {
  type                     = "ingress"
  from_port                = 5000
  to_port                  = 5000
  protocol                 = "tcp"
  security_group_id        = aws_security_group.mlflow_sg.id
  source_security_group_id = aws_security_group.predict_sg.id
  description              = "Allow predict to access MLflow on port 5000"
}
resource "aws_security_group_rule" "mlflow_ingress_api" {
  type                     = "ingress"
  from_port                = 5000
  to_port                  = 5000
  protocol                 = "tcp"
  security_group_id        = aws_security_group.mlflow_sg.id
  source_security_group_id = aws_security_group.api.id
  description              = "Allow API to access MLflow on port 5000"
}

# Security group for Train tasks
resource "aws_security_group" "train_sg" {
  name        = "mlops-train-sg"
  description = "Security group for Train ECS tasks"
  vpc_id      = aws_vpc.main.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for Predict tasks
resource "aws_security_group" "predict_sg" {
  name        = "mlops-predict-sg"
  description = "Security group for Predict ECS tasks"
  vpc_id      = aws_vpc.main.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- ECS SERVICE DISCOVERY (CLOUD MAP) FOR MLFLOW ---
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "mlops.local"
  description = "Private namespace for ECS services"
  vpc         = aws_vpc.main.id
}

resource "aws_service_discovery_service" "mlflow" {
  name = "mlflow-server"
  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      type = "A"
      ttl  = 10
    }
    routing_policy = "MULTIVALUE"
  }
  health_check_custom_config {}
}

# FastAPI ECS Task Definition (internal API)
resource "aws_iam_role" "fargate_task_execution" {
  name = "mlops-fargate-task-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

# CloudWatch log group for mlflow-server
resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/mlflow-server"
  retention_in_days = 7
}
# CloudWatch log group for predict
resource "aws_cloudwatch_log_group" "predict" {
  name              = "/ecs/predict"
  retention_in_days = 7
}
# CloudWatch log group for train
resource "aws_cloudwatch_log_group" "train" {
  name              = "/ecs/train"
  retention_in_days = 7
}
# CloudWatch log group for dashboard
resource "aws_cloudwatch_log_group" "dashboard" {
  name              = "/ecs/dashboard"
  retention_in_days = 7
}

# CloudWatch log group for FastAPI backend (internal API)
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/api"
  retention_in_days = 7
}
# Security group for internal FastAPI backend
resource "aws_security_group" "api" {
  name        = "mlops-api-sg"
  description = "Allow internal access to FastAPI backend"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
# FastAPI ECS Task Definition (internal API)
resource "aws_ecs_task_definition" "api" {
  family                   = "mlops-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.fargate_task_execution.arn
  task_role_arn            = aws_iam_role.fargate_task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/api:latest"
      essential = true
      portMappings = [{ containerPort = 8000, protocol = "tcp" }]
      environment = [
        { name = "AWS_S3_DATA_BUCKET", value = "${aws_s3_bucket.mlflow_artifacts.bucket}" },
        { name = "MLFLOW_TRACKING_URI", value = "http://mlflow-server.mlops.local:5000" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/api"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
  depends_on = [aws_cloudwatch_log_group.api]
}





resource "aws_iam_policy" "cloudwatch_logs" {
  name        = "CloudWatchLogsPolicy"
  description = "Allows log group creation and retention policy management"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = [
          "logs:CreateLogGroup",
          "logs:PutRetentionPolicy"
        ],
        Resource = "arn:aws:logs:*:*:log-group:/ecs/*"
      }
    ]
  })
}


# Policy for S3 and EFS access
resource "aws_iam_policy" "fargate_task_policy" {
  name = "mlops-fargate-task-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "elasticfilesystem:ClientMount",
          "elasticfilesystem:ClientWrite",
          "elasticfilesystem:ClientRootAccess"
        ]
        Resource = [aws_efs_file_system.mlflow.arn]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:CreateLogGroup"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "fargate_task_policy_attach" {
  role       = aws_iam_role.fargate_task_execution.name
  policy_arn = aws_iam_policy.fargate_task_policy.arn
}

# Attach AWS managed policy for ECS task execution
resource "aws_iam_role_policy_attachment" "ecs_task_execution_attach" {
  role       = aws_iam_role.fargate_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Fargate task/service skeletons (details to be filled in for each app)
# You will need to create ECR repos and push Docker images first

# Example for MLflow server (repeat for train, predict, dashboard)

# MLflow ECS Task Definition
resource "aws_ecs_task_definition" "mlflow" {
  family                   = "mlflow-server"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.fargate_task_execution.arn
  task_role_arn            = aws_iam_role.fargate_task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "mlflow-server"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/mlflow:latest"
      essential = true
      portMappings = [{ containerPort = 5000, protocol = "tcp" }]
      environment = [
        { name = "MLFLOW_BACKEND_STORE_URI", value = "file:///mnt/mlflow/mlruns" },
        { name = "AWS_S3_DATA_BUCKET", value = "${aws_s3_bucket.mlflow_artifacts.bucket}" }
      ]
      command = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--default-artifact-root", "s3://${aws_s3_bucket.mlflow_artifacts.bucket}/air_pollution_prediction"
      ]
      mountPoints = [
        {
          sourceVolume  = "mlflow-efs"
          containerPath = "/mnt/mlflow"
          readOnly      = false
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/mlflow-server"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:5000/ || exit 1"]
        interval    = 60
        timeout     = 15
        retries     = 3
        startPeriod = 90
      }
    }
  ])
  depends_on = [aws_cloudwatch_log_group.mlflow]

  volume {
    name = "mlflow-efs"
    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.mlflow.id
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      authorization_config {
        access_point_id = null
        iam             = "DISABLED"
      }
    }
  }
}

resource "aws_ecs_service" "mlflow" {
  name            = "mlflow-server"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.mlflow.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = [aws_subnet.private_a.id, aws_subnet.private_b.id]
    security_groups  = [aws_security_group.mlflow_sg.id]
    assign_public_ip = false
  }
  service_registries {
    registry_arn = aws_service_discovery_service.mlflow.arn
  }
}

# Training job ECS Task Definition
resource "aws_ecs_task_definition" "train" {
  family                   = "mlops-train"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.fargate_task_execution.arn
  task_role_arn            = aws_iam_role.fargate_task_execution.arn
  container_definitions = jsonencode([
    {
      name      = "train"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/train:latest"
      essential = true
      environment = [
        { name = "AWS_S3_DATA_BUCKET", value = "${aws_s3_bucket.mlflow_artifacts.bucket}" },
        { name = "MLFLOW_TRACKING_URI", value = "http://mlflow-server.mlops.local:5000" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/train"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
  depends_on = [aws_cloudwatch_log_group.train]
}


# Prediction job ECS Task Definition
resource "aws_ecs_task_definition" "predict" {
  family                   = "mlops-predict"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.fargate_task_execution.arn
  task_role_arn            = aws_iam_role.fargate_task_execution.arn
  container_definitions = jsonencode([
    {
      name      = "predict"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/predict:latest"
      essential = true
      environment = [
        { name = "AWS_S3_DATA_BUCKET", value = "${aws_s3_bucket.mlflow_artifacts.bucket}" },
        { name = "MLFLOW_TRACKING_URI", value = "http://mlflow-server.mlops.local:5000" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/predict"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
  depends_on = [aws_cloudwatch_log_group.predict]
}


# Dashboard ECS Task Definition
resource "aws_ecs_task_definition" "dashboard" {
  family                   = "mlops-dashboard"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.fargate_task_execution.arn
  task_role_arn            = aws_iam_role.fargate_task_execution.arn
  container_definitions = jsonencode([
    {
      name      = "dashboard"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/dashboard:latest"
      essential = true
      portMappings = [{ containerPort = 8501, protocol = "tcp" }]
      environment = [
        { name = "AWS_S3_DATA_BUCKET", value = "${aws_s3_bucket.mlflow_artifacts.bucket}" },
        { name = "API_BASE_URL", value = "http://${aws_lb.dashboard.dns_name}/api/v1" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/dashboard"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
  depends_on = [aws_cloudwatch_log_group.dashboard]
}



output "ecs_cluster_id" {
  value = aws_ecs_cluster.main.id
}

# --- SCHEDULED TASKS FOR TRAIN AND PREDICT ---

# IAM role for EventBridge to run ECS tasks
resource "aws_iam_role" "eventbridge_ecs" {
  name = "mlops-eventbridge-ecs-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "events.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "eventbridge_ecs" {
  name = "mlops-eventbridge-ecs-policy"
  role = aws_iam_role.eventbridge_ecs.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ecs:RunTask",
        "ecs:DescribeTasks",
        "iam:PassRole"
      ]
      Resource = "*"
    }]
  })
}

# Example: schedule train task every day at 2am UTC
resource "aws_cloudwatch_event_rule" "train_schedule" {
  name                = "mlops-train-schedule"
  //schedule_expression = "cron(0/20 * * * ? *)"
  schedule_expression = "cron(0 10 ? * THU *)"
}

resource "aws_cloudwatch_event_target" "train" {
  rule      = aws_cloudwatch_event_rule.train_schedule.name
  arn       = aws_ecs_cluster.main.arn
  role_arn  = aws_iam_role.eventbridge_ecs.arn
  ecs_target {
    task_definition_arn = aws_ecs_task_definition.train.arn
    launch_type         = "FARGATE"
    network_configuration {
      subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
      security_groups  = [aws_security_group.train_sg.id]
      assign_public_ip = true
    }
    platform_version = "1.4.0"
    enable_execute_command = true
  }
  depends_on = [aws_ecs_service.mlflow]
}

# Example: schedule predict task every day at 3am UTC
resource "aws_cloudwatch_event_rule" "predict_schedule" {
  name                = "mlops-predict-schedule"
  schedule_expression = "cron(0 * * * ? *)"
  //schedule_expression = "cron(0 0,6,12,18 * * ? *)"
}

resource "aws_cloudwatch_event_target" "predict" {
  rule      = aws_cloudwatch_event_rule.predict_schedule.name
  arn       = aws_ecs_cluster.main.arn
  role_arn  = aws_iam_role.eventbridge_ecs.arn
  ecs_target {
    task_definition_arn = aws_ecs_task_definition.predict.arn
    launch_type         = "FARGATE"
    network_configuration {
      subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
      security_groups  = [aws_security_group.predict_sg.id]
      assign_public_ip = true
    }
    platform_version = "1.4.0"
    enable_execute_command = true
  }
  depends_on = [aws_ecs_service.mlflow]
}

# --- PUBLIC LOAD BALANCER FOR DASHBOARD ---

resource "aws_lb" "dashboard" {
  name               = "mlops-dashboard-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]
  security_groups    = [aws_security_group.dashboard_alb.id]
}

resource "aws_lb_target_group" "dashboard" {
  name        = "mlops-dashboard-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  health_check {
    path                = "/"
    protocol            = "HTTP"
    matcher             = "200-399"
    interval            = 60
    timeout             = 15
    healthy_threshold   = 3
    unhealthy_threshold = 3
  }
}

# Target group for FastAPI API
resource "aws_lb_target_group" "api" {
  name        = "mlops-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  health_check {
    path                = "/api/v1/health"
    protocol            = "HTTP"
    matcher             = "200-399"
    interval            = 60
    timeout             = 15
    healthy_threshold   = 3
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "dashboard" {
  load_balancer_arn = aws_lb.dashboard.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.dashboard.arn
  }
}

resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.dashboard.arn
  priority     = 10
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
  condition {
    path_pattern {
      values = ["/api/v1/*"]
    }
  }
}

resource "aws_security_group" "dashboard_alb" {
  name        = "mlops-dashboard-alb-sg"
  description = "Allow HTTP access to dashboard ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Allow ALB to reach dashboard ECS task on port 8501

# Update dashboard ECS service to use ALB
resource "aws_ecs_service" "dashboard" {
  name            = "mlops-dashboard"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.dashboard.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups  = [aws_security_group.dashboard.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.dashboard.arn
    container_name   = "dashboard"
    container_port   = 8501
  }
  depends_on = [aws_lb_listener.dashboard]
}

output "dashboard_url" {
  value = aws_lb.dashboard.dns_name
  description = "Public URL for the dashboard."
}

# --- VPC ENDPOINTS FOR ECR AND S3 ---

resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.ecr.api"
  vpc_endpoint_type = "Interface"
  subnet_ids        = [aws_subnet.private_a.id, aws_subnet.private_b.id]
  security_group_ids = [
    aws_security_group.ecr_endpoint_sg.id
  ]
  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.ecr.dkr"
  vpc_endpoint_type = "Interface"
  subnet_ids        = [aws_subnet.private_a.id, aws_subnet.private_b.id]
  security_group_ids = [
    aws_security_group.ecr_endpoint_sg.id
  ]
  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [
    aws_route_table.private.id
  ]
}
