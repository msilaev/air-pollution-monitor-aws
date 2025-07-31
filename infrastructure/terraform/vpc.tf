
# Security group for CloudWatch Logs VPC endpoint
resource "aws_security_group" "cloudwatch_logs_endpoint_sg" {
  name        = "mlops-cloudwatch-logs-endpoint-sg"
  description = "Allow ECS tasks to access CloudWatch Logs endpoint on 443"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Allow HTTPS from ECS Task Security Groups"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    # IMPORTANT: List all security groups of your ECS tasks that send logs to CloudWatch
    security_groups = [
      aws_security_group.mlflow_sg.id,
      aws_security_group.train_sg.id,
      aws_security_group.predict_sg.id,
      aws_security_group.api.id,      # Assuming API also sends logs
      aws_security_group.dashboard.id # Assuming Dashboard also sends logs
    ]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mlops-cloudwatch-logs-endpoint-sg"
  }
}


# Security group for ECR endpoint
# Security group for ECR endpoint
resource "aws_security_group" "ecr_endpoint_sg" {
  name        = "mlops-ecr-endpoint-sg"
  description = "Allow ECS tasks in private subnets to access ECR endpoint on 443"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Allow HTTPS from ECS Task Security Groups"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    # IMPORTANT: List all security groups of your ECS tasks that need to pull images
    security_groups = [
      aws_security_group.mlflow_sg.id,
      aws_security_group.train_sg.id,
      aws_security_group.predict_sg.id,
      aws_security_group.api.id,      # Assuming API also pulls its image from ECR
      aws_security_group.dashboard.id # Assuming Dashboard also pulls its image from ECR
    ]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mlops-ecr-endpoint-sg"
  }
}

#Security Group for the CloudWatch Logs Endpoint

resource "aws_security_group" "logs_endpoint_sg" {
  name        = "mlops-logs-endpoint-sg"
  description = "Security group for CloudWatch Logs VPC Endpoint"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    # Allow ingress from all task security groups that will send logs
    security_groups = [
      aws_security_group.mlflow_sg.id,
      aws_security_group.train_sg.id,
      aws_security_group.predict_sg.id,
      aws_security_group.api.id,
      aws_security_group.dashboard.id
    ]
    description = "Allow ECS tasks to send logs to CloudWatch"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}


# Terraform configuration for AWS VPC, subnets, and basic networking



resource "aws_vpc" "main" {
  cidr_block = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "mlflow-vpc"
  }
}

resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_a_cidr
  availability_zone       = var.az_a
  map_public_ip_on_launch = true
  tags = {
    Name = "mlflow-public-a"
  }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_b_cidr
  availability_zone       = var.az_b
  map_public_ip_on_launch = true
  tags = {
    Name = "mlflow-public-b"
  }
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
  tags = {
    Name = "mlflow-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
  tags = {
    Name = "mlflow-public-rt"
  }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_a_id" {
  value = aws_subnet.public_a.id
}

output "public_subnet_b_id" {
  value = aws_subnet.public_b.id
}

# --- Variables for private subnets ---
variable "private_subnet_a_cidr" {
  description = "CIDR block for private subnet A"
  type        = string
}

variable "private_subnet_b_cidr" {
  description = "CIDR block for private subnet B"
  type        = string
}

# --- PRIVATE SUBNETS, NAT GATEWAY, AND ROUTING FOR ECS ---

resource "aws_subnet" "private_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.private_subnet_a_cidr
  availability_zone       = var.az_a
  map_public_ip_on_launch = false
  tags = {
    Name = "mlflow-private-a"
  }
}

resource "aws_subnet" "private_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.private_subnet_b_cidr
  availability_zone       = var.az_b
  map_public_ip_on_launch = false
  tags = {
    Name = "mlflow-private-b"
  }
}

# Elastic IP for NAT Gateway
#resource "aws_eip" "nat" {
  # vpc = true  # Removed: not supported in Terraform AWS provider >= v3.0
#}

# NAT Gateway in public subnet
#resource "aws_nat_gateway" "gw" {
#  allocation_id = aws_eip.nat.id
#  subnet_id     = aws_subnet.public_a.id
#  tags = {
#    Name = "mlflow-nat-gw"
#  }
#  depends_on = [aws_internet_gateway.gw]
#}

# Private route table
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  #route {
  #  cidr_block = "0.0.0.0/0"
  #  #nat_gateway_id = aws_nat_gateway.gw.id
  #}
  tags = {
    Name = "mlflow-private-rt"
  }
}

# Associate private subnets with private route table
resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private_b" {
  subnet_id      = aws_subnet.private_b.id
  route_table_id = aws_route_table.private.id
}

output "private_subnet_a_id" {
  value = aws_subnet.private_a.id
}

output "private_subnet_b_id" {
  value = aws_subnet.private_b.id
}
