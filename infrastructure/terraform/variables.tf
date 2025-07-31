variable "aws_region" {
  description = "AWS region to deploy resources in"
  default     = "eu-west-1"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  default     = "10.10.0.0/16"
}

variable "public_subnet_a_cidr" {
  description = "CIDR block for public subnet A"
  default     = "10.10.1.0/24"
}

variable "public_subnet_b_cidr" {
  description = "CIDR block for public subnet B"
  default     = "10.10.2.0/24"
}

variable "az_a" {
  description = "Availability zone for subnet A"
  default     = "eu-west-1a"
}

variable "az_b" {
  description = "Availability zone for subnet B"
  default     = "eu-west-1b"
}
